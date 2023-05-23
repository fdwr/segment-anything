// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { Tensor } from "onnxruntime-web";
import { modeDataProps } from "./Interfaces";

const modelData = ({ clicks, tensor, modelScale }: modeDataProps) => {
  const imageEmbedding = tensor;
  let pointCoords;
  let pointLabels;
  let pointCoordsTensor;
  let pointLabelsTensor;

  // Check there are input click prompts
  if (clicks) {
    let n = clicks.length;

    // If there is no box input, a single padding point with 
    // label -1 and coordinates (0.0, 0.0) should be concatenated
    // so initialize the array to support (n + 1) points.
    pointCoords = new Float32Array(2 * (n + 1));
    pointLabels = new Float32Array(n + 1);

    // Add clicks and scale to what SAM expects
    for (let i = 0; i < n; i++) {
      pointCoords[2 * i] = clicks[i].x * modelScale.samScale;
      pointCoords[2 * i + 1] = clicks[i].y * modelScale.samScale;
      pointLabels[i] = clicks[i].clickType;
    }

    // Add in the extra point/label when only clicks and no box
    // The extra point is at (0, 0) with label -1
    pointCoords[2 * n] = 0.0;
    pointCoords[2 * n + 1] = 0.0;
    pointLabels[n] = -1.0;

    // Create the tensor
    pointCoordsTensor = new Tensor("float32", pointCoords, [1, n + 1, 2]);
    pointLabelsTensor = new Tensor("float32", pointLabels, [1, n + 1]);
  }
  const imageSizeTensor = new Tensor("float32", [
    modelScale.height,
    modelScale.width,
  ]);

  if (pointCoordsTensor === undefined || pointLabelsTensor === undefined)
    return;

  // There is no previous mask, so default to an empty tensor
  const maskInput = new Tensor(
    "float32",
    new Float32Array(256 * 256),
    [1, 1, 256, 256]
  );
  // There is no previous mask, so default to 0
  const hasMaskInput = new Tensor("float32", [0]);

  return {
    image_embeddings: new Tensor(imageEmbedding.type, imageEmbedding.data.slice(0), imageEmbedding.dims),
    point_coords: pointCoordsTensor,
    point_labels: pointLabelsTensor,
    
    // orig_im_size is commented out because the optimized model is designed for a specific image
    // size. The WebNN EP is not mature enough yet to handle dynamic shapes based on the *values*
    // inside tensors, which requires more careful coordination/delegation of node assignments
    // because you don't want those values uploaded to the GPU only to perform a few tiny
    // operations on tensors that are only 2-4 elements in size, only to stall waiting to read
    // them back.
    // orig_im_size: imageSizeTensor,

    mask_input: maskInput,
    has_mask_input: hasMaskInput,
  };
};

export { modelData };
