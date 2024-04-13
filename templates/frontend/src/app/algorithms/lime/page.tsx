import React from "react";
import Head from "next/head";
import Image from "next/image";
import lime_img from "../../../assets/lime.png";

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
          <div className="mt-6 rounded-md shadow-xl ml-6 border-gray-700 border">
            <Image src={lime_img} width={500} height={500} alt="Lime image" />
          </div>
        </div>

        <div className="flex flex-row mt-4">
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
          <div></div>
        </div>
      </main>
    </div>
  );
};

export default lime;
