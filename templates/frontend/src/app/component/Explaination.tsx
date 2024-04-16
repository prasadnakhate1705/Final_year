import EXAI from "../../assets/EXAI.png";
import Image from "next/image";

const Explaination = () => {
  return (
    <div className="w-full lg:pl-16 lg:pr-10 px-16 flex flex-col sm:flex-row select-none justify-between">
      <div className="w-[40vw]">
        <h1 className="lg:text-5xl text-2xl font-semibold">
          From Motive to Method: Investigating the Why and How of AI Solutions.
        </h1>
        <p className="mt-2 font-mono">
          Model Performance Relies on Explainability as Much as Metrics.
          Understanding why and how a model behaves when errors occur is crucial
          for its reliability and trustworthiness.
        </p>
      </div>
      <div className="w-[50vw] mb-8">
        <Image src={EXAI} alt="EXAI_img" />
      </div>
    </div>
  );
};

export default Explaination;
