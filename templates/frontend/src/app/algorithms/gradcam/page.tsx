"use client";
import React from "react";
import Image from "next/image";
import gradcam_img from "../../../assets/gradcam.webp";
import { CopyBlock } from "react-code-blocks";
import { Copy } from "lucide-react";

const code = `import pandas as pd;
df = pd.read_csv('some_random.csv');
df.head(5)`;
const language = "python";
const myCustomTheme = {
  lineNumberColor: "#ccc",
  lineNumberBgColor: "#222",
  backgroundColor: "#222",
  textColor: "#ccc",
  substringColor: "#00ff00",
  keywordColor: "#0077ff",
  attributeColor: "#ffaa00",
  selectorTagColor: "#0077ff",
  docTagColor: "#aa00ff",
  nameColor: "#f8f8f8",
  builtInColor: "#0077ff",
  literalColor: "#ffaa00",
  bulletColor: "#ffaa00",
  codeColor: "#ccc",
  additionColor: "#00ff00",
  regexpColor: "#f8f8f8",
  symbolColor: "#ffaa00",
  variableColor: "#ffaa00",
  templateVariableColor: "#ffaa00",
  linkColor: "#aa00ff",
  selectorAttributeColor: "#ffaa00",
  selectorPseudoColor: "#aa00ff",
  typeColor: "#0077ff",
  stringColor: "#00ff00",
  selectorIdColor: "#ffaa00",
  quoteColor: "#f8f8f8",
  templateTagColor: "#ccc",
  deletionColor: "#ff0000",
  titleColor: "#0077ff",
  sectionColor: "#0077ff",
  commentColor: "#777",
  metaKeywordColor: "#f8f8f8",
  metaColor: "#aa00ff",
  functionColor: "#0077ff",
  numberColor: "#ffaa00",
};

const gradcam = () => {
  return (
    <div className="flex flex-col pt-10 min-h-screen py-2">
      <main className="flex flex-col w-full pt-20 flex-1 px-20 text-center">
        <div className="flex flex-row">
          <div>
            <h1 className="text-2xl font-bold text-left">LIME</h1>
            <p className="w-[50vw] text-left text-sm leading-loose">
              As a deep learning developer, understanding the inner workings of
              your models is crucial for building accurate and reliable systems.
              One important tool for this is model interpretation using GradCAM
              (Gradient Class Activation Maps). GradCAM is a technique that uses
              gradients to find which areas of an image are activated for a
              particular prediction, providing a visual explanation of the
              model's decision-making process.
            </p>
          </div>
          <div className="mt-6 rounded-lg shadow-xl ml-6 border-gray-300 border">
            <Image
              src={gradcam_img}
              width={500}
              height={500}
              className="mt-2"
              alt="Lime image"
            />
            <p className="italic text-sm text-gray-500 mt-4">
              Source: https://medium.com/@ninads79shukla/gradcam-73a752d368be
            </p>
          </div>
        </div>

        <div className="flex flex-row mt-4 justify-between">
          <div className="w-[50vw]">
            <p className="text-2xl text-left font-bold">STEPS: </p>
            <ul className="text-left text-sm leading-loose">
              <li>
                ➡{" "}
                <span className="font-semibold">
                  Load the Model and Preprocessors:
                </span>{" "}
                The first step in using Grad-CAM is to load the model and its
                preprocessors. This involves loading the saved weights of the
                model and any necessary preprocessing functions that were used
                during training.
              </li>
              <li>
                ➡{" "}
                <span className="font-semibold">
                  Find the Last Feature Map's Layer Name:
                </span>{" "}
                Identify last convolutional layer, typically preceding final
                activation layer, responsible for generating feature maps in the
                model architecture.
              </li>
              <li>
                ➡{" "}
                <span className="font-semibold">
                  Remove the Final Activation Layers:
                </span>{" "}
                Remove final activation layers, like softmax, as Grad-CAM
                operates on feature maps from the last convolutional layer.
              </li>
              <li>
                ➡{" "}
                <span className="font-semibold">
                  Calculate the Gradient Matrix:
                </span>{" "}
                Calculate gradient matrix for predicted class indices by
                computing gradients of predicted class scores with respect to
                feature maps.
              </li>
              <li>
                ➡{" "}
                <span className="font-semibold">
                  Generate Weighted Feature Map
                </span>{" "}
                Generate weighted feature map by converting gradient matrix to
                vector and multiplying with feature map from last convolutional
                layer.
              </li>{" "}
              <li>
                ➡{" "}
                <span className="font-semibold">
                  Resize Heatmap to Original Image Size:
                </span>{" "}
                Resize heatmap to original image size for visualization,
                providing context for important regions influencing model
                predictions.
              </li>
            </ul>
          </div>
          <div className="w-[35vw] text-sm text-left rounded-xl">
            <CopyBlock
              text={code}
              language={language}
              showLineNumbers={true}
              wrapLines={true}
              theme={myCustomTheme}
              codeBlock
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default gradcam;
