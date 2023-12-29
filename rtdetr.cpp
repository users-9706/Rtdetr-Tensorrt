#include <iostream>
#include <fstream>
#include <numeric>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include "common.h"
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include "NvOnnxParser.h"

using namespace std;
using namespace cv;
using namespace sample;
using namespace samplesCommon;


struct DetectRes {
	int classes;
	float x;
	float y;
	float w;
	float h;
	float prob;
};

int output_bbox_num;
const std::string output_blob_name = "output0";
const int det_bbox_len = 4;
const int det_cls_len = 80;


std::string engine_file = "./rtdetr-l.engine";
std::string labels_file = "";
int BATCH_SIZE = 1;
int INPUT_CHANNEL = 3;
int IMAGE_WIDTH = 640;
int IMAGE_HEIGHT = 640;
float obj_threshold = 0.4;
float nms_threshold = 0.45;

float conf_thresh = 0.5f;

int64 outSize;


nvinfer1::ICudaEngine* engine = nullptr;
nvinfer1::IExecutionContext* context = nullptr;

Logger gLogger{ sample::Logger::Severity::kINFO };


struct Object
{
	cv::Rect_<float> rect;
	int label;
	float score;
};

void draw_objects(const cv::Mat& image, const std::vector<Object>& objects)
{
	static const char* class_names[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};

	// cv::Mat image = bgr.clone();

	for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];

		fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.score,
			obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

		cv::Scalar color = cv::Scalar(0, 0, 255);
		float c_mean = cv::mean(color)[0];
		cv::Scalar txt_color;
		if (c_mean > 0.5) {
			txt_color = cv::Scalar(0, 0, 0);
		}
		else {
			txt_color = cv::Scalar(255, 255, 255);
		}

		cv::rectangle(image, obj.rect, color * 255, 2);

		char text[256];
		sprintf(text, "%s %.1f%%", class_names[obj.label], obj.score * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

		cv::Scalar txt_bk_color = color * 0.7 * 255;

		int x = obj.rect.x;
		int y = obj.rect.y + 1;
		//int y = obj.rect.y - label_size.height - baseLine;
		if (y > image.rows)
			y = image.rows;
		//if (x + label_size.width > image.cols)
			//x = image.cols - label_size.width;

		cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
			txt_bk_color, -1);

		cv::putText(image, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
	}
	cv::imshow("image", image);
	cv::waitKey(0);
}

void post_process_cpu(float* p, cv::Mat& src_img)
{

	float width = src_img.cols;
	float height = src_img.rows;
	float ratio_h = (640 * 1.0f) / height;
	float ratio_w = (640 * 1.0f) / width;
	//
	if (!src_img.data)
		return;
	float ratio = float(IMAGE_WIDTH) / float(src_img.cols) < float(IMAGE_HEIGHT) / float(src_img.rows) ? float(IMAGE_WIDTH) / float(src_img.cols) : float(IMAGE_HEIGHT) / float(src_img.rows);
	cv::Mat flt_img = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
	cv::Mat rsz_img;
	cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
	rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
	//
	
	
	std::vector<Object> proposals;

	for (int index = 0; index < output_bbox_num; index++)
	{
		float* ptr = p + index * (det_bbox_len + det_cls_len);
		float* pclass = ptr + det_bbox_len;

		int label = std::max_element(pclass, pclass + det_cls_len) - pclass;
		float confidence = pclass[label];

		if (confidence < conf_thresh) continue;

		float x_center = ptr[0];
		float y_center = ptr[1];
		float w = ptr[2];
		float h = ptr[3];

		float left = (x_center - w * 0.5f) * 640;
		float top = (y_center - h * 0.5f) * 640;
		float right = (x_center + w * 0.5f) * 640;
		float bottom = (y_center + h * 0.5f) * 640;

		Object obj;
		obj.rect = cv::Rect_<float>(left, top, right - left, bottom - top);

		obj.label = label;
		obj.score = confidence;
		proposals.push_back(obj);
	}

	draw_objects(flt_img, proposals);
}


std::vector<float> xywh2xyxy(std::vector<float> data) {
	float tl_x = data[0] - data[2] / 2.0;
	float tl_y = data[1] - data[3] / 2.0;
	float br_x = data[0] + data[2] / 2.0;
	float br_y = data[1] + data[3] / 2.0;
	return { tl_x, tl_y, br_x, br_y };

}


template <typename T>
vector<int> sort_indexes(vector<T>& v)
{
	vector<int> idx(v.size());
	iota(idx.begin(), idx.end(), 0);
	sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] > v[i2]; });	//这样是降序，改成<就是升序了
	return idx;
}
std::vector<float> prepareImage(std::vector<cv::Mat>& vec_img) {
	std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
	float* data = result.data();
	int index = 0;
	for (const cv::Mat& src_img : vec_img)
	{
		if (!src_img.data)
			continue;
		float ratio = float(IMAGE_WIDTH) / float(src_img.cols) < float(IMAGE_HEIGHT) / float(src_img.rows) ? float(IMAGE_WIDTH) / float(src_img.cols) : float(IMAGE_HEIGHT) / float(src_img.rows);
		cv::Mat flt_img = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
		cv::Mat rsz_img;
		cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
		rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
		flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255);
		flt_img = flt_img - cv::Scalar(0.485, 0.456, 0.406);
		int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
		std::vector<cv::Mat> split_img = {
				cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * (index + 2)),
				cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * (index + 1)),
				cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * index)
		};
		index += 3;
		cv::split(flt_img, split_img);
	}
	return result;
}


bool readTrtFile(const std::string& engineFile, //name of the engine file
	nvinfer1::ICudaEngine*& engine)
{
	std::string cached_engine;
	std::fstream file;
	std::cout << "loading filename from:" << engineFile << std::endl;
	nvinfer1::IRuntime* trtRuntime;
	file.open(engineFile, std::ios::binary | std::ios::in);

	if (!file.is_open()) {
		std::cout << "read file error: " << engineFile << std::endl;
		cached_engine = "";
	}

	while (file.peek() != EOF) {
		std::stringstream buffer;
		buffer << file.rdbuf();
		cached_engine.append(buffer.str());
	}
	file.close();

	trtRuntime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
	initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), ""); 
	engine = trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
	//std::cout << "deserialize done" << std::endl;

	return true;
}



float IOUCalculate(const DetectRes& det_a, const DetectRes& det_b) {
	cv::Point2f center_a(det_a.x, det_a.y);
	cv::Point2f center_b(det_b.x, det_b.y);
	cv::Point2f left_up(min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2), min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
	cv::Point2f right_down(max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
		max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
	float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
	float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
	float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
	float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
	float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
	float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
	if (inter_b < inter_t || inter_r < inter_l)
		return 0;
	float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
	float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
	if (union_area == 0)
		return 0;
	else
		return inter_area / union_area - distance_d / distance_c;
}

void LoadEngine() {
	// create and load engine
	std::fstream existEngine;
	existEngine.open(engine_file, std::ios::in);
	if (existEngine) {
		readTrtFile(engine_file, engine);
		assert(engine != nullptr);
	}
}


void EngineInference(const std::vector<std::string>& image_list, const int& outSize, void** buffers,
	const std::vector<int64_t>& bufferSize, cudaStream_t stream) {
	int index = 0;
	int batch_id = 0;
	std::vector<cv::Mat> vec_Mat(BATCH_SIZE);
	std::vector<std::string> vec_name(BATCH_SIZE);
	float total_time = 0;
	for (const std::string& image_name : image_list)
	{
		index++;
		std::cout << "Processing: " << image_name << std::endl;
		cv::Mat src_img = cv::imread(image_name);
		if (src_img.data)
		{
			//            cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
			vec_Mat[batch_id] = src_img.clone();
			vec_name[batch_id] = image_name;
			batch_id++;
		}
		if (batch_id == BATCH_SIZE || index == image_list.size())
		{
			auto t_start_pre = std::chrono::high_resolution_clock::now();
			std::cout << "prepareImage" << std::endl;
			std::vector<float>curInput = prepareImage(vec_Mat);
			auto t_end_pre = std::chrono::high_resolution_clock::now();
			float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
			std::cout << "prepare image take: " << total_pre << " ms." << std::endl;
			total_time += total_pre;
			batch_id = 0;
			if (!curInput.data()) {
				std::cout << "prepare images ERROR!" << std::endl;
				continue;
			}
			// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
			std::cout << "host2device" << std::endl;
			cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);

			// do inference
			std::cout << "execute" << std::endl;
			auto t_start = std::chrono::high_resolution_clock::now();
			context->execute(BATCH_SIZE, buffers);
			auto t_end = std::chrono::high_resolution_clock::now();
			float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
			std::cout << "Inference take: " << total_inf << " ms." << std::endl;
			total_time += total_inf;
			std::cout << "execute success" << std::endl;
			std::cout << "device2host" << std::endl;
			std::cout << "post process" << std::endl;
			auto r_start = std::chrono::high_resolution_clock::now();
			//auto *out = new float[outSize * BATCH_SIZE];
			void* p = malloc(outSize * BATCH_SIZE * sizeof(nvinfer1::DataType::kHALF));
			cudaMemcpyAsync(p, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);

			//cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
			cudaStreamSynchronize(stream);
			float* out = static_cast<float*>(p);

			post_process_cpu(out, src_img);
			//postProcessor(vec_Mat, out, outSize);


		}
	}
	std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}


bool InferenceFolder(const std::string& Image_path) {
	//std::vector<std::string> sample_images = readFolder(folder_name);
	std::vector<std::string> sample_images;
	sample_images.push_back(Image_path);
	//get context
	assert(engine != nullptr);
	context = engine->createExecutionContext();
	assert(context != nullptr);

	//get buffers
	assert(engine->getNbBindings() == 2);
	void* buffers[2];
	std::vector<int64_t> bufferSize;
	int nbBindings = engine->getNbBindings();
	bufferSize.resize(nbBindings);

	//
	int output_index= engine->getBindingIndex(output_blob_name.c_str());
	auto output_dims = engine->getBindingDimensions(output_index);
	output_bbox_num = output_dims.d[1];

	for (int i = 0; i < nbBindings; ++i) {
		nvinfer1::Dims dims = engine->getBindingDimensions(i);
		nvinfer1::DataType dtype = engine->getBindingDataType(i);
		int nLayers = engine->getNbLayers();
		int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
		bufferSize[i] = totalSize;
		std::cout << "binding" << i << ": " << totalSize << std::endl;
		cudaMalloc(&buffers[i], totalSize);
		if (i == 1) {
			outSize = totalSize / getElementSize(dtype);
		}
	}


	//get stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	//int outSize = bufferSize[1] / sizeof(float) / BATCH_SIZE;
	//int outSize = bufferSize[1] / 2 / BATCH_SIZE;

	EngineInference(sample_images, outSize, buffers, bufferSize, stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	cudaFree(buffers[0]);
	cudaFree(buffers[1]);

	// destroy the engine
	context->destroy();
	engine->destroy();

	//add
	return true;
}

void NmsDetect(std::vector<DetectRes>& detections) {
	sort(detections.begin(), detections.end(), [=](const DetectRes& left, const DetectRes& right) {
		return left.prob > right.prob;
		});

	for (int i = 0; i < (int)detections.size(); i++)
		for (int j = i + 1; j < (int)detections.size(); j++)
		{
			if (detections[i].classes == detections[j].classes)
			{
				float iou = IOUCalculate(detections[i], detections[j]);
				if (iou > nms_threshold)
					detections[j].prob = 0;
			}
		}

	detections.erase(std::remove_if(detections.begin(), detections.end(), [](const DetectRes& det)
		{ return det.prob == 0; }), detections.end());
}




int main()
{
	LoadEngine();

	string image_path = "./zidane.jpg";
	InferenceFolder(image_path);
}

