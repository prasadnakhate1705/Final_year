"use client";

import React from "react";
import { CardBody, CardContainer, CardItem } from "../../components/ui/3d-card";
import { useRouter } from "next/navigation";
import FileUpload from "../component/FileUpload";

const dashboard = () => {
  const router = useRouter();
  return (
    <div className="py-20">
      <div className="flex flex-col sm:flex-row justify-evenly">
        {/* MODEL_UPLOAD
        <div>
          <CardContainer className="inter-var">
            <CardBody className="bg-gray-50 relative group/card  border-black/[0.1] w-auto sm:w-[30rem] h-auto rounded-xl p-6 border  ">
              <CardItem
                translateZ="50"
                className="text-xl font-bold text-neutral-600 flex flex-row justify-between w-full"
              >
                <div>Upload your Model file</div>
                <div className="flex text-white bg-green-400 h-[20px] rounded-full text-xs font-semibold px-2">
                  In-Progress
                </div>
              </CardItem>
              <CardItem
                as="p"
                translateZ="60"
                className="text-neutral-500 text-sm max-w-sm mt-2 "
              >
                Try adding your custom built model here &darr;
              </CardItem>
              <CardItem translateZ="100" className="w-full mt-4"></CardItem>
              <div className="flex items-center justify-center mt-20">
                <CardItem
                  translateZ={20}
                  className="px-4 py-2 rounded-xl text-xs font-normal "
                ></CardItem>
              </div>
            </CardBody>
          </CardContainer>
        </div> */}
        {/* IMAGE_UPLOAD */}
        <div>
          <CardContainer className="inter-var">
            <CardBody className="bg-gray-50 relative group/card  border-black/[0.1] w-[350px] sm:w-[30rem] h-auto rounded-xl p-6 border  ">
              <CardItem
                translateZ="50"
                className="text-xl font-bold text-neutral-600 "
              >
                Upload your Image file
              </CardItem>
              <CardItem
                as="p"
                translateZ="60"
                className="text-neutral-500 text-sm max-w-sm mt-2 "
              >
                Try adding your X-ray image here &darr;
              </CardItem>
              <CardItem translateZ="100" className="w-full mt-4"></CardItem>
              <div className="flex items-center justify-center mt-20">
                <CardItem
                  translateZ={20}
                  className="px-4 py-2 rounded-xl text-xs font-normal"
                >
                  <FileUpload router={router} />
                </CardItem>
              </div>
            </CardBody>
          </CardContainer>
        </div>
      </div>
    </div>
  );
};

export default dashboard;
