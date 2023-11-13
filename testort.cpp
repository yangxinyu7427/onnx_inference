//
// Created by xyyang on 23-11-9.
//

#include <vector>
#include <iostream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>
#include <string>
#include <algorithm>
#include <map>
#include "onnxruntime_cxx_api.h"

using namespace std;
std::map<string,string> mockmap={
        {"position","1"},
        {"prop_country_id","219"},
        {"prop_starrating","3"},
        {"prop_brand_bool","1"},
        {"count_clicks","25"},
        {"count_bookings","20"},
        {"year","2013"},
        {"month","6"},
        {"weekofyear","24"},
        {"time","7"},
        {"site_id","5"},
        {"visitor_location_country_id","219"},
        {"srch_destination_id","13233"},
        {"srch_length_of_stay","2"},
        {"srch_booking_window","0"},
        {"srch_adults_count","2"},
        {"srch_children_count","0"},
        {"srch_room_count","1"},
        {"srch_saturday_night_bool","1"},
        {"random_bool","0"}
};

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t>& v) {
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}
// calculate size of tensor
int calculate_product(const std::vector<std::int64_t>& v) {
    int total = 1;
    for (auto& i : v) total *= i;
    return total;
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
    Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
    return tensor;
}

int main(int argc, char* argv[]) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;

    const char* model_path = "/home/xyyang/PycharmProjects/pythonProject/model_lr.onnx";

    Ort::Session session(env, model_path, session_options);
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;


//    vector<string> numerical_columns = {"prop_location_score1", "prop_location_score2", "prop_log_historical_price", "price_usd",
//            "orig_destination_distance", "prop_review_score", "avg_bookings_usd", "stdev_bookings_usd"};
//    vector<string> categorical_columns = {"position", "prop_country_id", "prop_starrating", "prop_brand_bool", "count_clicks",
//            "count_bookings", "year", "month", "weekofyear", "time", "site_id","visitor_location_country_id",
//            "srch_destination_id", "srch_length_of_stay", "srch_booking_window","srch_adults_count","srch_children_count",
//            "srch_room_count", "srch_saturday_night_bool","random_bool"};

    vector<std::string> input_names;
    vector<vector<std::int64_t>> input_shapes;
    vector<ONNXTensorElementDataType> input_types;
    cout << "Input Node Name :" << endl;
    for (size_t i = 0; i < session.GetInputCount(); i++) {
        input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());

        auto input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        for (auto& s : input_shape) {
            if (s < 0) {
                s = 1;
            }
        }
        input_shapes.emplace_back(input_shape);
        auto input_type = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        input_types.emplace_back(input_type);
        cout << i << "\t" << input_names.at(i) <<"\t" << print_shape(input_shape) <<"\t"<< input_type << endl;
    }

    vector<std::string> output_names;
    cout << "Output Node Name :" << endl;
    for (size_t i = 0; i < session.GetOutputCount(); i++) {
        output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        cout << i << "\t" << output_names.at(i)  << endl;
    }
    std::vector<Ort::Value> input_tensors;
    for(int i = 0; i < input_names.size(); i++){
        auto input_name=input_names[i];
        auto input_shape=input_shapes[i];
        auto total_number_elements = calculate_product(input_shape);

        switch (input_types[i]) {

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                {
                    // float
                    std::vector<float> input_tensor_values(total_number_elements);
                    std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return rand() % 2; });
                    input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_shape));
                    break;
                }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            {
                // int
                std::vector<int64_t> input_tensor_values(total_number_elements);
                std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return rand(); });
                input_tensors.emplace_back(vec_to_tensor<int64_t>(input_tensor_values, input_shape));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            {
                // string
                std::vector<string> input_tensor_values(total_number_elements);
                std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return mockmap[input_name]; });
                Ort::Value input_tensor = Ort::Value::CreateTensor(allocator, input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
                const char* const input_strings[] = {input_tensor_values[0].c_str()};
                input_tensor.FillStringTensor(input_strings, 1U);
                input_tensors.push_back(std::move(input_tensor));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
                break;
        }

    }


    // pass data through model
    std::vector<const char*> input_names_char(input_names.size(), nullptr);
    std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                   [&](const std::string& str) { return str.c_str(); });

    std::vector<const char*> output_names_char(output_names.size(), nullptr);
    std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                   [&](const std::string& str) { return str.c_str(); });

    std::cout << "Running model..." << std::endl;
    try {
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                          input_names_char.size(), output_names_char.data(), output_names_char.size());
        std::cout << "Done!" << std::endl;

        // double-check the dimensions of the output tensors
        // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
        assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());

    } catch (const Ort::Exception& exception) {
        std::cout << "ERROR running model inference: " << exception.what() << std::endl;
        exit(-1);
    }
    return 0;
}