// pages/result.tsx
"use client";
import React, { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { Button, buttonVariants } from "@/components/ui/button";
import cn from "../lib/utils";
import { FileCheck2 } from "lucide-react";

interface ResultProps {
  filename: string;
  result: string;
  hist_img: string;
  lime_exp_filename: string;
  grad_cam_imgs: { [key: string]: string };
  mfpp_exp: string;
}

const ResultPage: React.FC = () => {
  const [data, setData] = useState<ResultProps | null>({});

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("http://localhost:8080/get_global_data", {
          mode: "cors",
        });
        if (res.ok) {
          const result = await res.json(); // Parse the JSON response
          console.log(result);
          setData(result);
        } else {
          console.error("Error fetching data:", res.statusText);
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchData();
  }, []);

  // Check if data is null before destrcturing its properties
  if (!data) {
    return <div>Loading...</div>;
  }

  // Destructure data properties with optional chaining
  const {
    filename,
    result,
    hist_img,
    lime_exp_filename,
    grad_cam_imgs,
    mfpp_exp,
  } = data!;

  return (
    <div className="p-24 flex flex-col justify-center items-center font-semibold">
      <div className="sm:w-[800px] flex flex-col lg:flex-row gap-2 bg-gray-100 h-40 md:h-24 items-center justify-evenly rounded-lg w-[400px]">
        <h2 className="flex italic font-normal ml-2">
          <FileCheck2 />: {filename}
        </h2>
        <div className="flex flex-row">
          <div className="flex flex-row lg:flex-row items-center">
            <h2>Result: </h2> &nbsp;
            {result === "Normal" ? (
              <h2 className="font-semibold bg-green-300 text-white p-2 rounded-lg">
                {result}
              </h2>
            ) : (
              <h2 className="font-semibold bg-red-300 text-white p-2 rounded-lg">
                {result}
              </h2>
            )}
          </div>
          <Link href="/">
            <Button
              className={cn(
                buttonVariants(),
                "bg-blue-700 text-white ml-2 duration-300 font-bold items-center"
              )}
            >
              &larr; Back
            </Button>
          </Link>
        </div>
      </div>
      <Image
        src={`http://localhost:8080/static/${hist_img}`}
        alt="Histogram"
        width="800"
        height="800"
        className="shadow-xl p-4 rounded-lg"
      />
      <h2>LIME Explanation:</h2>
      <Image
        src={`http://localhost:8080/static/${lime_exp_filename}`}
        alt="LIME Explaination"
        width="200"
        height="200"
        className="shadow-xl p-4 rounded-lg"
      />
      <h2>MFPP Explanation</h2>
      <Image
        src={`http://localhost:8080/static/${mfpp_exp}`}
        alt="MFPP Explanation"
        width="800"
        height="800"
        className="shadow-xl p-4 rounded-lg"
      />
    </div>
  );
};

export default ResultPage;