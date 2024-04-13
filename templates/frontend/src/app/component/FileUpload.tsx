"use client";
import { useState } from "react";
import { Inbox } from "lucide-react";
import { ArrowRight, LoaderCircle } from "lucide-react";
import { Button, buttonVariants } from "../../components/ui/button";
import { toast } from "sonner";
import cn from "../lib/utils";

const FileUpload = ({ router }) => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      console.log(selectedFile);
    } else {
      setFile(null);
      setError("Please select an image file");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8080/", {
        method: "POST",
        body: formData,
      });
      console.log(response.json());
      if (!response.ok) {
        throw new Error("Failed to upload file");
      }

      // Handle success, e.g., display a success message
      console.log("File uploaded successfully");
      const d = new Date();

      //toast
      toast("Result has been generated!!", {
        description: d,
        action: {
          label: "✖",
          onClick: () => console.log("Undo"),
        },
      });

      setTimeout(router.push("/result"), 3000);
      // redirect
      // router.push("/result");
    } catch (error) {
      // Handle error, e.g., display an error message
      console.error("Error uploading file:", error.message);
    }
    setIsLoading(false);
  };

  return (
    <div className="relative">
      <form onSubmit={handleSubmit}>
        <div className="p-8 bg-white rounded-xl flex flex-row">
          <label
            htmlFor="inputTag"
            className="border-dashed border-4 cursor-pointer rounded-xl w-[300px] h-[100px] bg-gray-50 hover:bg-gray-100 py-4 flex justify-center items-center flex-col"
          >
            <>
              <Inbox className="w-10 h-10 text-blue-700 " />
              <p className="mt-2 text-sm text-slate-400 break-all">
                {file ? file.name : "Select your Image Here"}
              </p>
            </>
            <input
              type="file"
              name="file"
              accept="image/*"
              onChange={handleChange}
              className="absolute ml-4 mt-[10px] h-[100px]"
              id="inputTag"
              required
              style={{
                position: "absolute",
                width: "1px",
                height: "1px",
                padding: "0",
                margin: "-1px",
                overflow: "hidden",
                clip: "rect(0, 0, 0, 0)",
                border: "0",
              }}
            />
          </label>
          <button
            type="submit"
            onClick={() => file && setIsLoading(true)}
            className={cn(
              buttonVariants(),
              "bg-blue-700 text-white duration-300 rounded-xl h-[100px] ml-2 transition hover:scale-110 duration-300 ease-in-out"
            )}
          >
            {isLoading ? (
              <LoaderCircle className="animate-spin" />
            ) : (
              <ArrowRight />
            )}
          </button>
        </div>
      </form>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
};

export default FileUpload;
