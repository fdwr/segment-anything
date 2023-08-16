// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// npm install --g yarn
// yarn && yarn start
// http://localhost:8080/

import { InferenceSession, Tensor, env } from "onnxruntime-web";
import React, { useContext, useEffect, useState } from "react";
import "./assets/scss/App.scss";
import { handleImageScale } from "./components/helpers/scaleHelper";
import { modelScaleProps } from "./components/helpers/Interfaces";
import { onnxMaskToImage } from "./components/helpers/maskUtils";
import { modelData } from "./components/helpers/onnxModelAPI";
import Stage from "./components/Stage";
import AppContext from "./components/hooks/createContext";

/* @ts-ignore */
import npyjs from "npyjs";

// Define image, embedding and model paths
const IMAGE_PATH = "/assets/data/dogs.jpg";
const IMAGE_EMBEDDING = "/assets/data/dogs_embedding.npy";
// const MODEL_DIR = "/model/sam_onnx_example.onnx";
const MODEL_DIR = "/model/segment-anything-vit-h-static-shapes-origin-im-size-initializer-optimized-float32.onnx";

const App = () => {
  const {
    clicks: [clicks],
    image: [, setImage],
    maskImg: [, setMaskImg],
    executionTime: [, setExecutionTime]
  } = useContext(AppContext)!;
  const [model, setModel] = useState<InferenceSession | null>(null); // ONNX model
  const [tensor, setTensor] = useState<Tensor | null>(null); // Image embedding tensor
  // State to prevent next click injecting in to interfere the ongoing model running.
  // This only occurs in wasm.proxy = true (Web Worker) as which is asynchronous execution,
  // useEffect continously accepts new clicks that would cause race condition.
  const [isRunning, setIsRunning] = useState(false);
  // The ONNX model expects the input to be rescaled to 1024. 
  // The modelScale state variable keeps track of the scale values.
  const [modelScale, setModelScale] = useState<modelScaleProps | null>(null);

  // Initialize the ONNX model. load the image, and load the SAM
  // pre-computed image embedding
  useEffect(() => {
    // Initialize the ONNX model
    const initModel = async () => {

      env.wasm.numThreads = 1; // 4
      env.wasm.simd = true;
      env.wasm.proxy = true;
      // env.logLevel = "verbose"; //"error";
      // env.debug = true;

      const options: InferenceSession.SessionOptions = {
        // provider name: wasm, webnn
        // deviceType: cpu, gpu
        // powerPreference: default, high-performance

        // executionProviders: [{ name: "wasm"}], // WebAssembly CPU
        // executionProviders: [{ name: "webnn"}], // WebNN's default device (implementation defined)
        executionProviders: [{ name: "webnn", deviceType: "gpu", powerPreference: 'default' }],
        // logSeverityLevel: 0,
        // logVerbosityLevel: 3,
      };

      try {
        if (MODEL_DIR === undefined) return;
        const URL: string = MODEL_DIR;
        const model = await InferenceSession.create(URL, options);
        setModel(model);
      } catch (e) {
        console.log(e);
      }
    };
    initModel();

    // Load the image
    const url = new URL(IMAGE_PATH, location.origin);
    loadImage(url);

    // Load the Segment Anything pre-computed embedding
    Promise.resolve(loadNpyTensor(IMAGE_EMBEDDING, "float32")).then(
      (embedding) => setTensor(embedding)
    );
  }, []);

  const loadImage = async (url: URL) => {
    try {
      const img = new Image();
      img.src = url.href;
      img.onload = () => {
        const { height, width, samScale } = handleImageScale(img);
        setModelScale({
          height: height,  // original image height
          width: width,  // original image width
          samScale: samScale, // scaling factor for image which has been resized to longest side 1024
        });
        img.width = width; 
        img.height = height; 
        setImage(img);
      };
    } catch (error) {
      console.log(error);
    }
  };

  // Decode a Numpy file into a tensor. 
  const loadNpyTensor = async (tensorFile: string, dType: string) => {
    let npLoader = new npyjs();
    const npArray = await npLoader.load(tensorFile);
    const tensor = new Tensor(dType as Tensor.Type, npArray.data, npArray.shape);
    return tensor;
  };

  // Run the ONNX model every time clicks has changed
  useEffect(() => {
    if (!isRunning) {
      runONNX();
    }
  }, [clicks]);

  const runONNX = async () => {
    try {
      setIsRunning(true);
      if (
        model === null ||
        clicks === null ||
        tensor === null ||
        modelScale === null
      ) {
        setIsRunning(false);
        return;
      } else {
        // Prepare the model input in the correct format for SAM.
        // The modelData function is from onnxModelAPI.tsx.
        const feeds = modelData({
          clicks,
          tensor,
          modelScale,
        });
        if (feeds === undefined) return;
        // Run the SAM ONNX model with the feeds returned from modelData()
        const startTime = performance.now();
        const results = await model.run(feeds);
        const endTime = performance.now();
        const executionTime = endTime - startTime;
        setExecutionTime(executionTime);
        console.log(`model.run() took ${executionTime} ms`);
        const output = results[model.outputNames[0]];
        // The predicted mask returned from the ONNX model is an array which is 
        // rendered as an HTML image using onnxMaskToImage() from maskUtils.tsx.
        setMaskImg(onnxMaskToImage(output.data, output.dims[2], output.dims[3]));
        setIsRunning(false);
      }
    } catch (e) {
      console.log(e);
    }
  };

  return <Stage />;
};

export default App;
