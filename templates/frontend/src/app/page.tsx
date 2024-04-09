/* eslint-disable react/jsx-no-undef */
"use client";
import Link from "next/link";
import { ArrowRight, PlayCircle } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import cn from "./lib/utils";
import Image from "next/image";
import FileUpload from "./component/FileUpload";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();
  return (
    <>
      <div className="mb-12 mt-28 sm:mt-40 flex flex-col items-center justify-center text-center">
        <h1 className="max-w-4xl text-4xl md:text-5xl lg:text-5xl font-bold">
          Image <span className="text-blue-700">Detection</span>: Enhance
          Understanding with Explainable AI
        </h1>
        <p className="mt-5 max-w-prose text-zinc-700 sm:text-md">
          Images made simple: Go your way through complex Tuberculosis(TB)
          images with our explainable AI, with various models.
        </p>
        <div className="flex flex-row">
          <Link
            href="/dashboard"
            className={cn(
              buttonVariants(),
              "mt-5 bg-blue-700 text-white duration-300"
            )}
            target="_blank"
          >
            Get Started <ArrowRight className="ml-2 h-5 w-5" />
          </Link>
          <Link
            href="/dashboard"
            className={cn(
              buttonVariants(),
              "mt-5 bg-blue-700 text-white ml-2 duration-300"
            )}
            target="_blank"
          >
            See how it works{" "}
            <PlayCircle className="ml-2 h-5 w-5 animate-bounce" />
          </Link>
        </div>
        <FileUpload router={router} />
      </div>
    </>
  );
}