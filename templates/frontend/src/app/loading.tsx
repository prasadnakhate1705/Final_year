import { Loader2 } from "lucide-react";
import React from "react";

const LoadingAnimation: React.FC = () => {
  return (
    <div className="flex items-center justify-center h-screen">
      <Loader2 className="text-blue-700 animate-spin w-20 h-20 mr-2" />
    </div>
  );
};

export default LoadingAnimation;
