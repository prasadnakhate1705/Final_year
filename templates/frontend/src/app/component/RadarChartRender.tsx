import React from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";

const data = [
  {
    metrics: "Precision",
    model: 98.8587,
    full: 100,
  },
  {
    metrics: "Recall",
    model: 98.8571,
    full: 100,
  },
  {
    metrics: "F1",
    model: 98.8571,
    full: 100,
  },
];

const RadarChartRender = () => {
  return (
    <div style={{ width: "100%", height: "400px" }}>
      <ResponsiveContainer>
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="metrics" />
          <PolarRadiusAxis />
          <Radar
            name="Model"
            dataKey="model"
            stroke="#8884d8"
            fill="#8884d8"
            fillOpacity={0.6}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RadarChartRender;
