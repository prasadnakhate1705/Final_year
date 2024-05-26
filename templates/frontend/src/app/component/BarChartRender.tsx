import React from 'react';
import { BarChart, Bar, Rectangle, XAxis, YAxis, CartesianGrid, Tooltip, Legend} from 'recharts';

const data = [
  {
    name: 'Train 80%',
    normal: 2800,
    tuberculosis: 2800,
    amt: 5600,
  },
  {
    name: 'Test 10%',
    normal: 350,
    tuberculosis: 350,
    amt: 700,
  },
  {
    name: 'Validation 10%',
    normal: 350,
    tuberculosis: 350,
    amt: 700,
  },
];

const BarChartRender = () => {
  return (
      <BarChart
        width={500}
        height={400}
        data={data}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="normal" fill="#8884d8" activeBar={<Rectangle fill="pink" stroke="blue" />} />
        <Bar dataKey="tuberculosis" fill="#82ca9d" activeBar={<Rectangle fill="gold" stroke="purple" />} />
      </BarChart>
  );
};

export default BarChartRender;
