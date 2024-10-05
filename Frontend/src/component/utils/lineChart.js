import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const MyLineChart = ( apiData ) => {
  // Create the data structure for the line chart
  const data = apiData.content.predictions.soilMoisture.map((moisture, index) => ({
    name: `Day ${index + 1}`,
    value: moisture,
  }));

  return (
    <LineChart width={600} height={400} data={data}>
      <CartesianGrid stroke="#ccc" />
      <XAxis dataKey="name" />
      <YAxis />
      <Tooltip />
      <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} />
    </LineChart>
  );
};
export default MyLineChart;