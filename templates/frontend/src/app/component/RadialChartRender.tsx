import React from "react";
import { RadialBarChart, RadialBar, Legend } from "recharts";

interface Data {
  name: string;
  uv: number;
  fill: string;
}

const data: Data[] = [
  {
    name: "TP",
    uv: 2716,
    fill: "#8884d8",
  },
  {
    name: "FP",
    uv: 24,
    fill: "#83a6ed",
  },
  {
    name: "TN",
    uv: 2760,
    fill: "#8dd1e1",
  },
  {
    name: "FN",
    uv: 40,
    fill: "#82ca9d",
  },
];

const style = {
  top: "50%",
  right: 0,
  transform: "translate(0, -50%)",
  lineHeight: "24px",
};

const RadialChartRender = () => {
  return (
    <RadialBarChart
      cx="50%"
      cy="50%"
      innerRadius="30%"
      outerRadius="80%"
      barSize={20}
      data={data}
      height={400}
      width={400}
    >
      <RadialBar
        label={{ position: "insideStart", fill: "#000" }}
        background
        dataKey="uv"
      />
      <Legend
        iconSize={10}
        layout="vertical"
        verticalAlign="middle"
        wrapperStyle={style}
      />
    </RadialBarChart>
  );
};

export default RadialChartRender;
