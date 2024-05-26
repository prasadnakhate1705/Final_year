import Image from "next/image";
import l1 from "../../assets/l1.png";
import l2 from "../../assets/l2.png";
import l3 from "../../assets/l3.png";

const Features = () => {
  return (
    <div className="w-full px-10 lg:px-16 py-4 flex flex-col lg:flex-row gap-[20px] select-none">
      <div className="rounded-lg p-2 py-2 cursor-pointer hover:shadow-xl w-[500px] sm:w-[413px] sm:h-[120px] border border-gray-300 hover:border-blue-700 duration-300 transition hover:-translate-y-1 flex flex-row">
        <div className="mt-6 ml-10">
          <Image src={l1} width={120} height={40} alt="Picture of the author" />
        </div>
        <div className="ml-6">
          <p className="text-md font-semibold text-black">
            Deploy with integrity
          </p>
          <p className="text-sm text-gray-500 font-sans antialized">
            Provide clear insights into the tuberculosis detection model's
            decision-making process, building trust in its results.
          </p>
        </div>
      </div>
      <div className="rounded-lg p-2 py-2 cursor-pointer hover:shadow-xl w-[500px] sm:w-[413px] sm:h-[120px] border border-gray-300 hover:border-blue-700 duration-300 transition hover:-translate-y-1 flex flex-row">
        <div className="mt-6 ml-10">
          <Image src={l2} width={100} height={40} alt="Picture of the author" />
        </div>
        <div className="ml-6">
          <p className="text-md font-semibold text-black">
            Promote Transparency
          </p>
          <p className="text-sm text-gray-500 font-sans antialized">
            Document the AI model's lifecycle to enhance trust in its
            tuberculosis detection capabilities.
          </p>
        </div>
      </div>
      <div className="rounded-lg p-2 py-2 cursor-pointer hover:shadow-xl w-[500px] sm:w-[413px] sm:h-[120px] border border-gray-300 hover:border-blue-700 duration-300 transition hover:-translate-y-1 flex flex-row">
        <div className="mt-6 ml-10">
          <Image src={l3} width={130} height={40} alt="Picture of the author" />
        </div>
        <div className="ml-6">
          <p className="text-md font-semibold text-black">Drive Engagement</p>
          <p className="text-sm text-gray-500 font-sans antialized">
            Offer an innovative platform for tuberculosis detection that
            educates and informs users, driving collaboration in healthcare
            decisions.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Features;
