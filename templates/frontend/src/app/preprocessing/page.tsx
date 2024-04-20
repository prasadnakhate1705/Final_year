"use client";
import React from "react";
import Image from "next/image";
import { useState } from "react";
import normal from "../../assets/data-preprocessing/normal.jpg";
import tb from "../../assets/data-preprocessing/tb.jpg";
import histogram from "../../assets/data-preprocessing/histogram.png";
import augment from "../../assets/data-preprocessing/augment.jpg";
import augment_diff from "../../assets/data-preprocessing/augment_diff.png";

const Page = () => {
  const [activeItem, setActiveItem] = useState("cs");

  const handleItemClick = (item: React.SetStateAction<string>) => {
    setActiveItem(item);
  };
  return (
    <div className="p-10 py-20 sm:p-28 items-center justify-center grid grid-col">
      {/* transformation */}
      <div className="text-left text-sm w-[920px]">
        <h1 className="text-xl font-bold font-quicksand text-white bg-blue-700 rounded-md w-fit p-1 ">
          Data Preprocessing
        </h1>{" "}
        &nbsp;
        <p>
          <strong>Transformation:</strong> <br />
          <strong>Objective:</strong> (Tabulating) Converting{" "}
          <strong>Image.PNG</strong> Files to <strong>NumPy Arrays</strong>
        </p>
        <div className="w-full align-self self-center mt-2">
          <Image src={normal} alt="normal" className="rounded-xl shadow-2xl" />
          &nbsp;
          <Image src={tb} alt="tb" className="rounded-xl shadow-2xl" />
        </div>
      </div>

      {/* Explaination histogram */}
      <div className="text-left text-sm w-[920px] mt-8">
        <h1 className="text-xl font-bold text-white bg-blue-700 rounded-md w-fit p-1 ">
          Explaination Histogram
        </h1>{" "}
        &nbsp;
        <p>
          <strong>Pixel Intensity:</strong> Ranges from 0 (completely black) to
          255 (completely white).
        </p>
        <p>
          <strong>X-axis:</strong> Represents the entire datasets darkness
          (closer to 0) to brightness (closer to 255).
        </p>
        <strong>Slope of the Red Line:</strong>
        <ul>
          <li>
            In a Right-Skewed Distribution, the mean pixel intensity is higher,
            indicating brighter areas.
          </li>
          <li>
            In a Left-Skewed Distribution, the mean pixel intensity is lower,
            signifying darker areas.
          </li>
        </ul>
        <p>Therefore,</p>
        <ul>
          <li>
            <strong>
              X<sub>Tuberulosis</sub>:
            </strong>{" "}
            Contains brighter or whiter areas.
          </li>
          <li>
            <strong>
              X<sub>Normal</sub>:
            </strong>{" "}
            Contains darker areas.
          </li>
        </ul>
        <div className="w-full align-self self-center mt-4">
          <Image
            src={histogram}
            alt="histogram"
            className="rounded-xl shadow-2xl"
          />
        </div>
      </div>

      {/* Data Augmentation */}
      <div className="text-left text-sm w-[920px] mt-8">
        <h1 className="text-xl font-bold text-white bg-blue-700 rounded-md w-fit p-1">
          Data Augmentation (Class imbalance problem)
        </h1>{" "}
        <p className="mt-4">
          Perform data augmentation techniques such as rotation, scaling,
          flipping, and adding noise to increase the diversity and size of your
          training dataset. This helps in improving the models generalization
          ability.
        </p>
        <strong>Objective:</strong> (Tabulating) Converting solve class
        imbalance problem (700 TB images and 3500 normal images)
        <h3 className="font-bold">Not Recommended:</h3>
        <ol className="list-decimal ml-4">
          <li>
            <strong>Reflection:</strong>
            <ul className=" ml-4">
              <li>
                Reflection in x-axis is discouraged due to potential confusion
                (e.g., 6 vs. 9).
              </li>
              <li>
                Reflection in y-axis results in non-physiologic images, not
                recommended.
              </li>
              <li>
                No method simulates differences between Posterior-Anterior (PA)
                and Anteroposterior (AP) chest X-rays.
              </li>
            </ul>
          </li>
          <li>
            <strong>Severe Rotation:</strong>
            <ul className=" ml-4">
              <li>
                Avoid severe rotations (−90 to 90) as they may introduce
                unrealistic noise.
              </li>
              <li>
                Slight rotations (−5 to 5) seen in practice can be helpful.
              </li>
            </ul>
          </li>
          <li>
            <strong>Scaling:</strong>
            <ul className=" ml-4">
              <li>
                Large scaling (×1) stretches the image, while small scaling (×1)
                reduces its size.
              </li>
              <li>
                Equal scaling in both axes is possible; scaling in only one axis
                is not recommended clinically.
              </li>
            </ul>
          </li>
          <li>
            <strong>Shearing:</strong>
            <ul className=" ml-4">
              <li>
                Shearing is not recommended, producing clinically non-existent
                images.
              </li>
            </ul>
          </li>
        </ol>
        <div className="w-full align-self self-center mt-4">
          <Image
            src={augment}
            alt="data augmentation"
            className="rounded-xl shadow-2xl"
          />
          &nbsp;
          <Image
            src={augment_diff}
            alt="data augmentation_diff"
            className="rounded-xl shadow-2xl"
          />
        </div>
      </div>

      {/* Image Enhancement */}
      <div className="flex flex-col text-sm w-[920px] mt-8">
        <h1 className="text-xl font-bold text-white bg-blue-700 rounded-md w-fit p-1 ">
          Image Enhancement
        </h1>{" "}
        &nbsp;
        <nav className="flex flex-row">
          <ul className="flex flex-row w-[920px] gap-3 justify-center items-center">
            <li
              className={`justify-self-center border-2 hover:border-b-blue-700 rounded-lg duration-200 p-2 ${
                activeItem === "cs" ? "border-b-blue-700" : ""
              }`}
              onClick={() => handleItemClick("cs")}
            >
              Contrast Stretching
            </li>
            <li
              className={`justify-self-center border-2 hover:border-b-blue-700 rounded-lg duration-200 p-2 ${
                activeItem === "he" ? "border-b-blue-700" : ""
              }`}
              onClick={() => handleItemClick("he")}
            >
              Histogram Equalization
            </li>
            <li
              className={`justify-self-center border-2 hover:border-b-blue-700 rounded-lg duration-200 p-2 ${
                activeItem === "ahe" ? "border-b-blue-700" : ""
              }`}
              onClick={() => handleItemClick("ahe")}
            >
              Adaptive Histogram Equalization
            </li>
          </ul>
        </nav>
        <div className="mt-10">
          {activeItem === "cs" && <div> Contrast Stretching</div>}
          {activeItem === "he" && <div>Histogram Equalization</div>}
          {activeItem === "ahe" && <div>Adaptive Histogram Equalization</div>}
        </div>
      </div>
    </div>
  );
};

export default Page;
