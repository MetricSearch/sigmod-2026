#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // Auto-convert std::vector
#include <pybind11/numpy.h>     // Optional: for numpy arrays
#include "rabitqlib/quantization/rabitq.hpp"
#include "bindings_common.hpp"

namespace py = pybind11;
namespace rabitqlib::python_bindings {
    class Quantiser {
    public:
        Quantiser() : f_add(0), f_rescale(0), f_error(0) {}
        py::tuple quantise_single(
           py::array_t<float> data,
           py::array_t<float> centroid,
           int dim,
           int bits,
           bool rotate
        ){
            auto data_buf = data.request();
            auto centroid_buf = centroid.request();
            float* data_ptr = static_cast<float*>(data_buf.ptr);
            float* centroid_ptr = static_cast<float*>(centroid_buf.ptr);
            std::vector<uint32_t> code(dim);  // code
            
            if (rotate == true) {
                rabitqlib::Rotator<float>* rotator = rabitqlib::choose_rotator<float>(dim);
                size_t padded_dim = rotator->size();
                std::vector<float> rotated_data(padded_dim);
                rotator->rotate(data_ptr, rotated_data.data());

                rabitqlib::quant::quantize_full_single(
                    rotated_data.data(),
                    centroid_ptr,
                    dim,
                    bits,
                    code.data(),
                    f_add,
                    f_rescale,
                    f_error,
                    rabitqlib::METRIC_L2
                );
                return py::make_tuple(code, rotated_data);
            }else{
                rabitqlib::quant::quantize_full_single(
                    data_ptr,
                    centroid_ptr,
                    dim,
                    bits,
                    code.data(),
                    f_add,
                    f_rescale,
                    f_error,
                    rabitqlib::METRIC_L2
                );
            return py::make_tuple(code, data);
            }
        }  

    private:
        float f_add =0;                      // factors for estimating similarity
        float f_rescale =0;                  // factors for estimating similarity
        float f_error =0;                    // factors for computing error bounds
    };
}  // namespace rabitqlib::python_bindings

void register_quantiser(py::module_& m) {
    using namespace rabitqlib::python_bindings;

    py::class_<Quantiser>(m, "Quantiser")
        .def(py::init<>())
        .def(
           "quantise_single",
           &Quantiser::quantise_single,
           py::arg("data"),
           py::arg("centroid"),
           py::arg("dim"),
           py::arg("bits")=2,
           py::arg("rotate")=false
        );
}