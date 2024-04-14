"use client";
import React from "react";
import Image from "next/image";
import lime_img from "../../../assets/lime.png";
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

const lime = () => {
  return (
    <div className="flex flex-col pt-10 min-h-screen py-2">
      <main className="flex flex-col w-full pt-20 flex-1 pl-20 pr-14 text-center">
        <div className="flex flex-row">
          <div>
            <h1 className="text-2xl font-bold text-left">LIME</h1>
            <p className="w-[50vw] text-sm text-left leading-loose">
              LIME, known as Local Interpretable Model-agnostic Explanations, is
              a method designed to shed light on the predictions made by machine
              learning models. It operates independently of the model type,
              focusing on explaining predictions in a localized context to
              ensure accuracy in reflecting the model's behavior around a
              specific data point. To create explanations, LIME alters the input
              data around the point of interest and constructs an interpretable
              model using this perturbed data to explain the model's
              predictions. This interpretable model, often a simplified linear
              model, is then used to provide insights into the prediction based
              on the original input features.
            </p>
          </div>
          <div className="mt-6 rounded-lg shadow-xl ml-6 border-gray-300 border">
            <Image src={lime_img} width={500} height={500} alt="Lime image" />
            <p className="italic text-sm text-gray-500 mt-6">
              Source: https://arxiv.org/pdf/1602.04938.pdf
            </p>
          </div>
        </div>

        <div className="flex flex-row mt-4 justify-between">
          <div className="w-[50vw]">
            <p className="text-2xl text-left font-bold">WORKING: </p>
            <ul className="text-left text-sm leading-loose">
              <li>
                ➡ <span className="font-semibold">Perturbing the input:</span>{" "}
                LIME perturbs the input around the instance to be explained,
                creating a set of perturbed data points.
              </li>
              <li>
                ➡ <span className="font-semibold">Obtaining predictions:</span>{" "}
                The original model's predictions are obtained for the perturbed
                data points.
              </li>
              <li>
                ➡{" "}
                <span className="font-semibold">
                  Weighting perturbed data points:
                </span>{" "}
                The perturbed data points are weighted based on their proximity
                to the original instance.
              </li>
              <li>
                ➡{" "}
                <span className="font-semibold">
                  Learning an interpretable model:
                </span>{" "}
                An interpretable model, such as a sparse linear model, is
                learned on the weighted perturbed data points and their
                associated predictions.
              </li>
              <li>
                ➡{" "}
                <span className="font-semibold">Generating explanations:</span>{" "}
                The interpretable model is used to generate explanations for the
                prediction in terms of the original input features.
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

export default lime;
