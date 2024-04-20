import React from "react";
import Image from "next/image";
import normal from "../../assets/data-preprocessing/normal.jpg";
import tb from "../../assets/data-preprocessing/tb.jpg";
import histogram from "../../assets/data-preprocessing/histogram.png";

const page = () => {
  return (
    <div className="p-10 py-20 sm:p-28 items-center justify-center grid grid-col">

      {/* transformation */}
      <div className="text-left text-sm w-[920px]">
        <h1 className="text-xl font-bold font-quicksand text-white bg-blue-700 rounded-lg w-fit p-1 ">Data Preprocessing</h1> &nbsp;
        <p>
          <strong>Transformation:</strong> <br />
          <strong>Objective:</strong> (Tabulating) Converting{" "}
          <strong>Image.PNG</strong> Files to <strong>NumPy Arrays</strong>
        </p>
      </div>
      <div className="w-full align-self self-center mt-2">
        <Image src={normal} alt="normal" className="rounded-xl shadow-2xl" />
        &nbsp;
        <Image src={tb} alt="tb" className="rounded-xl shadow-2xl" />
      </div>

      {/* Explaination histogram */}
      <div className="text-left text-sm w-[920px] mt-8">
      <h1 className="text-xl font-bold text-white bg-blue-700 rounded-lg w-fit p-1 ">Explaination Histogram</h1> &nbsp;
        <p>
          <strong>Pixel Intensity:</strong> Ranges from 0 (completely black) to 255 (completely white).
        </p>
        <p>
          <strong>X-axis:</strong> Represents the entire datasets darkness
          (closer to 0) to brightness (closer to 255).
        </p>
          <strong>Slope of the Red Line:</strong>
          <ul>
            <li>
              In a Right-Skewed Distribution, the mean pixel intensity is higher, indicating brighter areas.
            </li>
            <li>
              In a Left-Skewed Distribution, the mean pixel intensity is lower, signifying darker areas.
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
    </div>
  );
};

export default page;
