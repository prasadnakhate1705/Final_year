// pages/result.tsx
"use client";
import React, { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { Button, buttonVariants } from "@/components/ui/button";
import cn from "../lib/utils";
import { FileCheck2 } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import axios from "axios";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../../components/ui/dialog";
import { Loader2 } from "lucide-react";

interface ResultProps {
  filename: string;
  result: string;
  hist_img: string;
  lime_exp_filename: string;
  grad_cam_imgs: { [key: string]: string };
  mfpp_exp: string;
  predictionBymodel: string;
  model_stats: {
    Model: string;
    accuracy_train: string;
    accuracy_val: string;
    Subset: string;
    "Training time": string;
    "Training in seconds": number;
    TP: number;
    FP: number;
    FN: number;
    TN: number;
    Precision: number;
    Recall: number;
    F1: number;
    AUC: number;
    Accuracy: number;
  };
  get_config: {
    name: string;
    weight_decay: null | number;
    clipnorm: null | number;
    global_clipnorm: null | number;
    clipvalue: null | number;
    use_ema: boolean;
    ema_momentum: number;
    ema_overwrite_frequency: null | number;
    jit_compile: boolean;
    is_legacy_optimizer: boolean;
    learning_rate: number;
    beta_1: number;
    beta_2: number;
    epsilon: number;
    amsgrad: boolean;
  };
  lrp_image: string;
}

const ResultPage: React.FC = () => {
  const [data, setData] = useState<ResultProps | null>();
  const [modelExplainationText, setModelExplainationText] =
    useState<string>("");
  const lime_text = "LIME(Local Interpretable Model-Agnostic Explanations)";

  const model_exp = useMutation({
    mutationFn: async () => {
      const response = await axios.post("/api/model_exp", {
        model_text: lime_text,
      });
      return response.data;
    },
  });

  const infoClick = () => {
    model_exp.mutate(undefined, {
      onSuccess: (data) => {
        console.log("Actual data fetched in UI", modelExplainationText);
        setModelExplainationText(data);
      },
      onError: (error) => {
        console.log(error);
        window.alert("failed to create model_Explaiantions");
      },
    });
  };

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

  const {
    filename,
    result,
    hist_img,
    lime_exp_filename,
    grad_cam_imgs,
    mfpp_exp,
    predictionBymodel,
    model_stats,
    get_config,
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
      <div className="flex gap-2">
        <Image
          src={`http://localhost:8080/static/${lime_exp_filename}`}
          alt="LIME Explaination"
          width="200"
          height="200"
          className="shadow-xl p-4 rounded-lg"
        />
        <Dialog>
          <DialogTrigger asChild>
            <Button
              className="bg-blue-700 w-2 h-2 p-2 mt-4 rounded-full text-white shadow-lg animate-pulse"
              onClick={infoClick}
            >
              i
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-white justify-center items-center text-left">
            <DialogHeader className="justify-center items-center">
              <DialogTitle>LIME</DialogTitle>
            </DialogHeader>
            <pre className="bg-gary-300 text-black text-wrap">
              {model_exp.isPending ? (
                <Loader2 className="text-blue-700 animate-spin w-20 h-20 mr-2" />
              ) : (
                modelExplainationText
              )}
            </pre>
          </DialogContent>
        </Dialog>
      </div>
      <h2>MFPP Explanation</h2>
      <Image
        src={`http://localhost:8080/static/${mfpp_exp}`}
        alt="MFPP Explanation"
        width="800"
        height="800"
        className="shadow-xl p-4 rounded-lg"
      />
      <div>
        <h2>Grad-CAM Images:</h2>
        <div className="flex flex-wrap">
          {grad_cam_imgs &&
            Object.keys(grad_cam_imgs).map((layer) => (
              <div key={layer} className="m-2">
                <h3>{layer}</h3>
                <Image
                  src={`http://localhost:8080/static/${grad_cam_imgs[layer]}`}
                  alt={`${layer} Grad-CAM`}
                  width="200"
                  height="200"
                  className="shadow-xl p-4 rounded-lg"
                />
              </div>
            ))}
        </div>
      </div>
      <div className="p-24 flex flex-col justify-center items-center font-semibold">
        {/* Existing code */}

        {/* Render predictionBymodel */}
        <h2>Prediction By Model:</h2>
        <p>{predictionBymodel}</p>

        {/* Render model_stats attributes individually */}
        <div className="shadow-xl rounded-lg p-4">
          <h2>Model Stats:</h2>
          <ul>
            <li>
              <strong>Model:</strong> {model_stats?.Model}
            </li>
            <li>
              <strong>Accuracy (Training):</strong>{" "}
              {model_stats?.accuracy_train}
            </li>
            <li>
              <strong>Accuracy (Validation):</strong>{" "}
              {model_stats?.accuracy_val}
            </li>
            <li>
              <strong>Subset:</strong> {model_stats?.Subset}
            </li>
            <li>
              <strong>Training Time:</strong> {model_stats?.["Training time"]}
            </li>
            <li>
              <strong>Training Time (Seconds):</strong>{" "}
              {model_stats?.["Training in seconds"]}
            </li>
            {/* Render other attributes similarly */}
          </ul>
        </div>

        {/* Render get_config attributes individually */}
        <div className="shadow-xl rounded-lg p-4">
          <h2>Get Config:</h2>
          <ul>
            <li>
              <strong>Name:</strong> {get_config?.name}
            </li>
            <li>
              <strong>Weight Decay:</strong> {get_config?.weight_decay}
            </li>
            <li>
              <strong>Clipnorm:</strong> {get_config?.clipnorm}
            </li>
            <li>
              <strong>Global Clipnorm:</strong> {get_config?.global_clipnorm}
            </li>
            {/* Render other attributes similarly */}
          </ul>
        </div>
      </div>
      <div className="shadow-xl rounded-lg">
        <h2>LRP</h2>
        <Image
          src={`http://localhost:8080/static/${lrp_image}`}
          alt="LRP"
          width="800"
          height="800"
          className="shadow-xl p-4 rounded-lg"
        />
      </div>
    </div>
  );
};

export default ResultPage;
