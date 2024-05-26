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
import BarChartRender from "../component/BarChartRender";
import PieChartRender from "../component/PieChartRender";
import RadialChartRender from "../component/RadialChartRender";

interface ResultProps {
  filename: string;
  result: string;
  hist_img: string;
  lime_exp_filename: string;
  lung_seg: string;
  grad_cam_imgs: { [key: string]: string };
  mfpp_exp: string;
  predictionBymodel: string;
  model_stats: {
    Model: string;
    accuracy_train: number;
    accuracy_val: number;
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
  const [analytics, setAnalytics] = useState<ResultProps | "hidden">("hidden");
  const [modelExplainationText, setModelExplainationText] =
    useState<string>("");

  const model_exp = useMutation({
    mutationFn: async (filename: string) => {
      const response = await axios.post("/api/model_exp", {
        img_path: filename,
      });
      return response.data;
    },
  });

  const infoClick = (filename: string) => {
    model_exp.mutate(filename, {
      onSuccess: (data) => {
        console.log("Actual data fetched in UI", modelExplainationText);
        setModelExplainationText(data);
      },
      onError: (error) => {
        console.log(error);
        window.alert("failed to create model_Explainations");
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

  const toggleAnalytics = () => {
    setAnalytics((prevState) => (prevState === "hidden" ? data : "hidden"));
  };

  return (
    <div className="p-24 flex flex-col font-semibold gap-8">
      <div className="grid grid-cols-2 flex-row gap-[30rem] items-center justify-center">
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
            <Link href="/dashboard">
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
        <div className="">
          <Button onClick={toggleAnalytics}>Model Information &darr;</Button>
        </div>
      </div>
      {analytics != "hidden" && (
        <div className={`p-18 flex flex-col font-semibold gap-4`}>
          <h2 className="text-center bg-blue-700 rounded-sm text-white w-fit p-2">
            Model Prediction Statistics:{" "}
          </h2>
          <div className="shadow-xl rounded-lg p-4 grid grid-cols-2">
            <BarChartRender />
            <div className="text-center">
              <PieChartRender
                accuracy_train={model_stats?.accuracy_train}
                color={"#8884d8"}
                name={"Train"}
              />
              <h2>Accuracy_Training Set</h2>
            </div>
            <div className="text-center">
              <PieChartRender
                accuracy_train={model_stats?.accuracy_val}
                color={"#923454"}
                name={"Val"}
              />
              <h2>Accuracy_Validation Set</h2>
            </div>
            <div className="ml-28">
              <RadialChartRender />
              <h2 className="ml-24">Accuracy_Validation Set</h2>
            </div>
          </div>
          {/* Render predictionBymodel */}
          <h2 className="bg-blue-700 rounded-sm text-white w-fit p-2">
            Prediction By Model:
          </h2>
          <p>{predictionBymodel}</p>
        </div>
      )}
      <Image
        src={`http://localhost:8080/static/${hist_img}`}
        alt="Histogram"
        width="800"
        height="800"
        className="shadow-xl p-4 rounded-lg"
      />
      <div className="grid grid-cols-2 w-fit">
        <div className="flex flex-col gap-2">
          <div className="flex flex-row gap-2">
            <h2 className="bg-blue-700 rounded-sm text-white w-fit p-2">
              LIME Explanation:
            </h2>
            <Dialog>
              <DialogTrigger asChild>
                <Button
                  className="bg-blue-700 w-4 h-4 p-3 mt-2 rounded-full text-white shadow-lg animate-pulse"
                  onClick={() => infoClick(data.lung_seg)}
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
          <Image
            src={`http://localhost:8080/static/lung_seg_gen.png`}
            alt="LIME Explaination"
            width="300"
            height="300"
            className="shadow-xl p-4 rounded-lg"
          />
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex flex-row gap-2">
            <h2 className="bg-blue-700 rounded-sm text-white w-fit p-2">
              LRP Explanation
            </h2>
            <Dialog>
              <DialogTrigger asChild>
                <Button
                  className="bg-blue-700 w-4 h-4 p-3 mt-2 rounded-full text-white shadow-lg animate-pulse"
                  onClick={() => infoClick(data.lrp_image)}
                >
                  i
                </Button>
              </DialogTrigger>
              <DialogContent className="bg-white justify-center items-center text-left">
                <DialogHeader className="justify-center items-center">
                  <DialogTitle>LRP</DialogTitle>
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
          <Image
            src={`http://localhost:8080/static/conv2d_57_lrp.png`}
            alt="LRP Explaination"
            width="400"
            height="400"
            className="shadow-xl p-4 rounded-lg"
          />
        </div>
      </div>
      <div className="flex flex-row gap-2">
        <h2 className="bg-blue-700 rounded-sm text-white w-fit p-2">
          MFPP Explanation
        </h2>
        <Dialog>
          <DialogTrigger asChild>
            <Button
              className="bg-blue-700 w-4 h-4 p-3 mt-2 rounded-full text-white shadow-lg animate-pulse"
              onClick={() => infoClick(data.mfpp_exp)}
            >
              i
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-white justify-center items-center text-left">
            <DialogHeader className="justify-center items-center">
              <DialogTitle>MFPP</DialogTitle>
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
      <Image
        src={`http://localhost:8080/static/${mfpp_exp}`}
        alt="MFPP Explanation"
        width="800"
        height="800"
        className="shadow-xl p-4 rounded-lg"
      />
      <div>
        <div className="flex flex-row gap-2">
          <h2 className="bg-blue-700 rounded-sm text-white w-fit p-2">
            Grad-CAM Images:
          </h2>
          <Dialog>
            <DialogTrigger asChild>
              <Button
                className="bg-blue-700 w-4 h-4 p-3 mt-2 rounded-full text-white shadow-lg animate-pulse"
                onClick={() => infoClick(data.grad_cam_imgs[0])}
              >
                i
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-white justify-center items-center text-left">
              <DialogHeader className="justify-center items-center">
                <DialogTitle>MFPP</DialogTitle>
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
    </div>
  );
};

export default ResultPage;
