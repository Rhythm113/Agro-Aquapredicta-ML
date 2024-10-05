import React from 'react';
import './barChart.css';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const MyBarChart = (apiData) => {
    console.log(apiData)
    const data = apiData.content.predictions.soilMoisture.map((moisture, index) => ({
    name: `Day ${index + 1}`, // X-axis labels: Day 1, Day 2, etc.
    value: moisture,          // Y-axis values: soil moisture
  }));

  return (
    <BarChart width={600} height={300} data={data}>
      <CartesianGrid stroke="#ccc" />
      <XAxis dataKey="name" />
      <YAxis />
      <Tooltip />
      <Bar dataKey="value" fill="#3B621D" />
    </BarChart>
  );
};
export default MyBarChart;