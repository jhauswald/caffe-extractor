/*
 *  Copyright (c) 2015, University of Michigan.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

/**
 * @author: Johann Hauswald, Yiping Kang
 * @contact: jahausw@umich.edu, ypkang@umich.edu
 */
#include <assert.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include "boost/program_options.hpp"
#include "caffe/caffe.hpp"

using namespace std;
using namespace cv;

using caffe::Blob;
using caffe::Net;
using caffe::Caffe;

namespace po = boost::program_options;
po::variables_map parse_opts(int ac, char** av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")(
      "network,n", po::value<string>()->default_value("imc.prototxt"),
      "Network config file (.prototxt)")(
      "weights,w", po::value<string>()->default_value("imc.caffemodel"),
      "Pretrained weights (.caffemodel)")(
      "tolayer,l", po::value<int>()->default_value(0),
      "Pretrained weights (.caffemodel)")(
      "image,i", po::value<string>()->default_value("test.jpg"),
      "input image(jpeg)");

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return vm;
}

int main(int argc, char** argv) {
  po::variables_map vm = parse_opts(argc, argv);

  int tolayer = vm["tolayer"].as<int>();
  string network = vm["network"].as<string>();
  string weights = vm["weights"].as<string>();
  string image = vm["image"].as<string>();

  Net<float> *net = new Net<float>(network);
  net->CopyTrainedLayersFrom(weights);
  Caffe::set_mode(Caffe::CPU);

  Mat img;
  LOG(INFO) << "Reading " << image;
  img = imread(image);

  if (img.channels() * img.rows * img.cols != net->input_blobs()[0]->count())
    LOG(FATAL) << "Incorrect " << image << ", resize to correct dimensions.\n";

  // prepare data into array
  float *data = (float*)malloc(img.channels() * img.rows * img.cols * sizeof(float));

  int pix_count = 0;
  for (int c = 0; c < img.channels(); ++c) {
    for (int i = 0; i < img.rows; ++i) {
      for (int j = 0; j < img.cols; ++j) {
        Vec3b pix = img.at<Vec3b>(i, j);
        float* p = (float*)(data);
        p[pix_count] = pix[c];
        ++pix_count;
      }
    }
  }

  vector<Blob<float>*> in_blobs = net->input_blobs();
  in_blobs[0]->set_cpu_data((float*)data);
  std::ofstream ofs ("input.out");
  LOG(INFO) << "Dumping input data";

  for(int d = 0; d < net->blobs()[0]->count(); ++d)
    ofs << net->blobs()[0]->cpu_data()[d] << "\n";

  for(int i = 0; i <= tolayer; ++i) {
    net->ForwardTo(i);
    string filename = "layer" + to_string(i) + ".out";
    std::ofstream ofs(filename.c_str());
    ofs << net->blobs()[i]->num() << " "
        << net->blobs()[i]->channels() << " "
        << net->blobs()[i]->height() << " "
        << net->blobs()[i]->width() << endl;

    for(int d = 0; d < net->blobs()[i]->count(); ++d)
      ofs << net->blobs()[1]->cpu_data()[d] << "\n";
  }

  free(net);
  free(data);

  return 0;
}
