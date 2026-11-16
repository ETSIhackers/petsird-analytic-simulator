// petsird_fast: read / write the time blocks of PETSIRD (binary) files from / to flat arrays
//
// The (small) header is read with the yardl-generated Python package. This extension
// reads all time blocks with the yardl-generated C++ code (libpetsird) and returns the
// events as one contiguous array per field (struct of arrays), such that Python never
// handles individual events or time blocks.
//
// All arrays are 1D/2D C-contiguous with fixed dtypes and are handed over to Python
// without copying (the std::vector is owned by a capsule). Other array frameworks
// (e.g. cupy, pytorch on GPU) can be supported later via nanobind's DLPack support.

#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <string>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "petsird/binary/protocols.h"

namespace nb = nanobind;

namespace {

template <typename T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& values, std::vector<size_t> shape) {
  auto* owned = new std::vector<T>(std::move(values));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T>(owned->data(), shape.size(), shape.data(), owner);
}

template <typename T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& values) {
  size_t n = values.size();
  return to_numpy(std::move(values), {n});
}

bool is_identity(const petsird::RigidTransformation& t) {
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 4; ++j)
      if (std::abs(t.matrix(i, j) - (i == j ? 1.0f : 0.0f)) > 1e-6f) return false;
  return true;
}

// coincidences of one stream (prompts or delayeds) of one module type pair
struct CoincidenceArrays {
  std::vector<uint32_t> det_bin0, det_bin1, tof_idx;
  // number of events before every event time block (+ total): the events of event time
  // block k are [block_offsets[k], block_offsets[k+1])
  std::vector<int64_t> block_offsets{0};

  void append(const petsird::ListOfCoincidenceEvents& events) {
    for (const auto& e : events) {
      det_bin0.push_back(e.detection_bins[0]);
      det_bin1.push_back(e.detection_bins[1]);
      tof_idx.push_back(e.tof_idx);
    }
  }

  void end_time_block() { block_offsets.push_back(static_cast<int64_t>(det_bin0.size())); }

  nb::dict to_dict() {
    nb::dict d;
    d["det_bin0"] = to_numpy(std::move(det_bin0));
    d["det_bin1"] = to_numpy(std::move(det_bin1));
    d["tof_idx"] = to_numpy(std::move(tof_idx));
    d["block_offsets"] = to_numpy(std::move(block_offsets));
    return d;
  }
};

class TimeBlockReader {
 public:
  explicit TimeBlockReader(const std::string& fname) : reader_(fname) {
    reader_.ReadHeader(header_);
    num_module_types_ = header_.scanner.scanner_geometry.replicated_modules.size();
  }

  size_t num_module_types() const { return num_module_types_; }
  bool done() const { return done_; }

  // read up to max_time_blocks time blocks (0: all remaining ones); if start_ms / stop_ms
  // are given, only time blocks whose time interval is within [start_ms, stop_ms] are
  // converted (the others are counted in num_skipped_time_blocks)
  nb::dict read(size_t max_time_blocks, std::optional<uint32_t> start_ms,
                std::optional<uint32_t> stop_ms) {
    std::map<std::pair<size_t, size_t>, CoincidenceArrays> prompts, delayeds;
    for (size_t m1 = 0; m1 < num_module_types_; ++m1)
      for (size_t m2 = 0; m2 <= m1; ++m2) {
        prompts[{m1, m2}];
        delayeds[{m1, m2}];
      }
    std::vector<uint32_t> event_start, event_stop;
    // singles histograms per detection bin, one flat vector per module type
    std::vector<std::vector<uint64_t>> singles(num_module_types_);
    std::vector<size_t> singles_size(num_module_types_, 0);
    std::vector<uint32_t> singles_start, singles_stop;
    std::map<std::string, int64_t> counts;

    size_t num_read = 0, num_skipped = 0;
    auto in_range = [&](const petsird::TimeInterval& t) {
      return (!start_ms || t.start >= *start_ms) && (!stop_ms || t.stop <= *stop_ms);
    };
    {
      // the GIL is not needed while reading (no Python objects are touched)
      nb::gil_scoped_release release;
      petsird::TimeBlock time_block;
      while (!done_ && (max_time_blocks == 0 || num_read < max_time_blocks)) {
        if (!reader_.ReadTimeBlocks(time_block)) {
          reader_.Close();
          done_ = true;
          break;
        }
        ++num_read;
        const auto& interval = std::visit([](const auto& b) -> const petsird::TimeInterval& { return b.time_interval; }, time_block);
        if (!in_range(interval)) {
          ++num_skipped;
          continue;
        }
        if (auto* tb = std::get_if<petsird::EventTimeBlock>(&time_block)) {
          event_start.push_back(tb->time_interval.start);
          event_stop.push_back(tb->time_interval.stop);
          for (auto& [stream, arrays] :
               {std::pair{&tb->prompt_events, &prompts}, std::pair{&tb->delayed_events, &delayeds}}) {
            for (size_t m1 = 0; m1 < stream->size(); ++m1)
              for (size_t m2 = 0; m2 < (*stream)[m1].size() && m2 <= m1; ++m2)
                (*arrays)[{m1, m2}].append((*stream)[m1][m2]);
            for (auto& [pair, a] : *arrays) a.end_time_block();
          }
          for (const auto& s : tb->single_events) counts["single_events"] += s.size();
          for (const auto& a : tb->triple_events)
            for (const auto& b : a)
              for (const auto& c : b) counts["triple_events"] += c.size();
          for (const auto& a : tb->quadruple_events)
            for (const auto& b : a)
              for (const auto& c : b)
                for (const auto& d : c) counts["quadruple_events"] += d.size();
        } else if (auto* tb = std::get_if<petsird::SinglesHistogramTimeBlock>(&time_block)) {
          if (tb->singles_per_detection_bin_histogram.empty()) {
            counts["singles_unit_histogram_time_blocks"] += 1;
            continue;
          }
          if (tb->singles_per_detection_bin_histogram.size() != num_module_types_)
            throw std::runtime_error("singles histogram does not have one entry per module type");
          singles_start.push_back(tb->time_interval.start);
          singles_stop.push_back(tb->time_interval.stop);
          for (size_t m = 0; m < num_module_types_; ++m) {
            const auto& h = tb->singles_per_detection_bin_histogram[m];
            if (singles_start.size() == 1)
              singles_size[m] = h.size();
            else if (h.size() != singles_size[m])
              throw std::runtime_error("singles histograms of different sizes");
            singles[m].insert(singles[m].end(), h.begin(), h.end());
          }
        } else if (auto* tb = std::get_if<petsird::BedMovementTimeBlock>(&time_block)) {
          counts[is_identity(tb->transform) ? "identity_bed_movement_time_blocks"
                                            : "bed_movement_time_blocks"] += 1;
        } else if (auto* tb = std::get_if<petsird::GantryMovementTimeBlock>(&time_block)) {
          bool identity = true;
          for (const auto& t : tb->transforms) identity = identity && is_identity(t);
          counts[identity ? "identity_gantry_movement_time_blocks"
                          : "gantry_movement_time_blocks"] += 1;
        } else if (std::holds_alternative<petsird::WeightedEventTimeBlock>(time_block)) {
          counts["weighted_event_time_blocks"] += 1;
        } else if (std::holds_alternative<petsird::DeadTimeTimeBlock>(time_block)) {
          counts["dead_time_time_blocks"] += 1;
        } else if (std::holds_alternative<petsird::ExternalSignalTimeBlock>(time_block)) {
          counts["external_signal_time_blocks"] += 1;
        }
      }
    }

    nb::dict result, prompts_dict, delayeds_dict, singles_dict, counts_dict;
    for (auto& [pair, a] : prompts) prompts_dict[nb::make_tuple(pair.first, pair.second)] = a.to_dict();
    for (auto& [pair, a] : delayeds) delayeds_dict[nb::make_tuple(pair.first, pair.second)] = a.to_dict();
    nb::list singles_counts;
    size_t num_singles_blocks = singles_start.size();
    for (size_t m = 0; m < num_module_types_; ++m)
      singles_counts.append(
          to_numpy(std::move(singles[m]), {num_singles_blocks, singles_size[m]}));
    singles_dict["counts"] = singles_counts;
    singles_dict["start"] = to_numpy(std::move(singles_start));
    singles_dict["stop"] = to_numpy(std::move(singles_stop));
    for (auto& [name, n] : counts) counts_dict[name.c_str()] = n;

    result["prompts"] = prompts_dict;
    result["delayeds"] = delayeds_dict;
    result["event_time_blocks"] = nb::dict();
    result["event_time_blocks"]["start"] = to_numpy(std::move(event_start));
    result["event_time_blocks"]["stop"] = to_numpy(std::move(event_stop));
    result["singles"] = singles_dict;
    result["other"] = counts_dict;
    result["num_time_blocks"] = num_read;
    result["num_skipped_time_blocks"] = num_skipped;
    return result;
  }

 private:
  petsird::binary::PETSIRDReader reader_;
  petsird::Header header_;
  size_t num_module_types_ = 0;
  bool done_ = false;
};

// writes time blocks from flat arrays; the header is read from a PETSIRD file (written
// with the yardl-generated Python code, see petsird_fast.TimeBlockWriter)
class TimeBlockWriter {
 public:
  using Indices = nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  using Counts = nb::ndarray<const uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

  TimeBlockWriter(const std::string& fname, const std::string& header_fname) : writer_(fname) {
    petsird::binary::PETSIRDReader header_reader(header_fname);
    header_reader.ReadHeader(header_);
    petsird::TimeBlock unused;
    while (header_reader.ReadTimeBlocks(unused)) {
    }
    header_reader.Close();
    num_module_types_ = header_.scanner.scanner_geometry.replicated_modules.size();
    writer_.WriteHeader(header_);
  }

  // prompts / delayeds: {(type_1, type_2): (det_bin0, det_bin1, tof_idx)} (uint32 arrays);
  // delayeds None: no delayed events in this time block (empty list)
  void add_event_time_block(uint32_t start, uint32_t stop, nb::dict prompts, nb::object delayeds) {
    petsird::EventTimeBlock tb;
    tb.time_interval.start = start;
    tb.time_interval.stop = stop;
    fill(tb.prompt_events, prompts);
    if (!delayeds.is_none()) fill(tb.delayed_events, nb::cast<nb::dict>(delayeds));
    pending_.emplace_back(std::move(tb));
  }

  // counts: singles counts of all detection bins, one uint64 array per module type
  void add_singles_histogram_time_block(uint32_t start, uint32_t stop, nb::list counts) {
    petsird::SinglesHistogramTimeBlock tb;
    tb.time_interval.start = start;
    tb.time_interval.stop = stop;
    for (auto c : counts) {
      auto a = nb::cast<Counts>(c);
      tb.singles_per_detection_bin_histogram.emplace_back(a.data(), a.data() + a.shape(0));
    }
    pending_.emplace_back(std::move(tb));
  }

  // write all added time blocks (one at a time, which gives the same file as the
  // yardl-generated Python writer)
  void flush() {
    if (pending_.empty()) return;
    {
      nb::gil_scoped_release release;
      for (const auto& tb : pending_) writer_.WriteTimeBlocks(tb);
    }
    pending_.clear();
  }

  void close() {
    if (closed_) return;
    flush();
    writer_.EndTimeBlocks();
    writer_.Close();
    closed_ = true;
  }

 private:
  void fill(petsird::LowerTriangularMatrix<petsird::ListOfCoincidenceEvents>& matrix, nb::dict events) {
    matrix.resize(num_module_types_);
    for (size_t m1 = 0; m1 < num_module_types_; ++m1) matrix[m1].resize(m1 + 1);
    for (auto [key, value] : events) {
      auto pair = nb::cast<std::pair<size_t, size_t>>(key);
      if (pair.first >= num_module_types_ || pair.second > pair.first)
        throw std::invalid_argument("module type pairs have to be (type_1, type_2) with type_2 <= type_1");
      auto arrays = nb::cast<std::tuple<Indices, Indices, Indices>>(value);
      const auto& [b0, b1, tof] = arrays;
      size_t n = b0.shape(0);
      if (b1.shape(0) != n || tof.shape(0) != n)
        throw std::invalid_argument("det_bin0, det_bin1 and tof_idx have different lengths");
      auto& list = matrix[pair.first][pair.second];
      list.resize(n);
      for (size_t i = 0; i < n; ++i) {
        list[i].detection_bins = {b0.data()[i], b1.data()[i]};
        list[i].tof_idx = tof.data()[i];
      }
    }
  }

  petsird::binary::PETSIRDWriter writer_;
  petsird::Header header_;
  size_t num_module_types_ = 0;
  std::vector<petsird::TimeBlock> pending_;
  bool closed_ = false;
};

}  // namespace

NB_MODULE(_petsird_fast, m) {
  m.doc() = "fast reading of PETSIRD time blocks into flat arrays (yardl C++ / libpetsird)";
  nb::class_<TimeBlockReader>(m, "TimeBlockReader")
      .def(nb::init<const std::string&>(), nb::arg("fname"),
           "open a PETSIRD binary file and read its header (C++)")
      .def("read", &TimeBlockReader::read, nb::arg("max_time_blocks") = 0,
           nb::arg("start_ms").none() = nb::none(), nb::arg("stop_ms").none() = nb::none(),
           "read up to max_time_blocks time blocks (0: all remaining ones); with start_ms / "
           "stop_ms, only time blocks with a time interval within [start_ms, stop_ms] are "
           "converted")
      .def_prop_ro("num_module_types", &TimeBlockReader::num_module_types)
      .def_prop_ro("done", &TimeBlockReader::done);
  nb::class_<TimeBlockWriter>(m, "TimeBlockWriter")
      .def(nb::init<const std::string&, const std::string&>(), nb::arg("fname"),
           nb::arg("header_fname"),
           "create a PETSIRD binary file with the header of the PETSIRD file header_fname")
      .def("add_event_time_block", &TimeBlockWriter::add_event_time_block, nb::arg("start"),
           nb::arg("stop"), nb::arg("prompts"), nb::arg("delayeds").none() = nb::none())
      .def("add_singles_histogram_time_block", &TimeBlockWriter::add_singles_histogram_time_block,
           nb::arg("start"), nb::arg("stop"), nb::arg("counts"))
      .def("flush", &TimeBlockWriter::flush, "write all added time blocks as one batch")
      .def("close", &TimeBlockWriter::close);
}
