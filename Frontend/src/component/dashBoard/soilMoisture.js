import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import "./soilMoisture.css";
import CircleProgressBar from "../utils/circleProgressBar";

const SoilMoisture = () => {
  const navigate = useNavigate();
  const { t } = useTranslation(); // Access translations

  // State to hold selected date
  const [selectedDate, setSelectedDate] = useState("");

  // Get today's date in "YYYY-MM-DD" format
  const getTodayDate = () => {
    const today = new Date();
    const yyyy = today.getFullYear();
    const mm = String(today.getMonth() + 1).padStart(2, "0"); // Months are 0-based
    const dd = String(today.getDate()).padStart(2, "0");
    return `${yyyy}-${mm}-${dd}`;
  };
  const dateConverter = (data)=>{
    const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const dateString = data;
        const datePart = dateString.split(' ')[0];
        const dateObj = new Date(datePart);
        let day = dateObj.getDate();
        day = day < 10 ? '0'+day : day
        
        const monthIndex = dateObj.getMonth();
        const year = dateObj.getFullYear();

        const formattedDate = day + '/' + monthNames[monthIndex] + '/' + year;
        return formattedDate
    }
  useEffect(() => {
    // Set default date as today
    setSelectedDate(dateConverter(getTodayDate()));
  }, []);

  const soilMoistureData = [
    {
      totalFields: 25,
      avgPercentage: 75,
      title: "High",
    },
    {
      totalFields: 8,
      avgPercentage: 55,
      title: "Optimal",
    },
    {
      totalFields: 45,
      avgPercentage: 25,
      title: "Low",
    },
  ];

  // Function to handle date input change
  const handleDateChange = (e) => {
    setSelectedDate(dateConverter(e.target.value));
  };

  return (
    <div className="soilMoistureSection" onClick={()=>{navigate("/soil")}}>
      <div className="soilMoistureHeading">
        <h3>Soil Moisture</h3>
        <div>
          {/* Display "Today" if today's date is selected */}
          <label htmlFor='dateInput' className="dateSpan">{selectedDate === getTodayDate() ? "Today" : selectedDate}</label>
          <input id="dateInput"
            type="date"
            value={selectedDate}
            onChange={handleDateChange}
          />
        </div>
      </div>
      <div className="soilMoistureContentBox">
        {soilMoistureData.map((data) => (
          <div key={data.title} className="progressContent">
            <h5 className="fieldNum">{data.totalFields} Fields</h5>
            <CircleProgressBar content={data} />
            <h4>{data.title}</h4>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SoilMoisture;
