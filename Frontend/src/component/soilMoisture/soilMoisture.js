import React, { useEffect, useState, useRef, Fragment } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import "./soilMoisture.css";
import Header from "../layout/header";
import MyBarChart from "../utils/barChart";
import MyPieChart from "../utils/pieChart";

const SoilMoistureDetails = () => {
    const navigate = useNavigate();
    const { t } = useTranslation(); // Access translations
    const apiData = {
        predictions: {
          soilMoisture: [30, 32, 34, 36, 38, 35, 33]
        },
        // graphs: {
        //   soilMoisture: [30, 35, 40, 38, 33, 28, 34, 37, 30, 25, 40, 38],
        //   vegetationHealth: [75, 80, 85, 88, 82, 78, 84, 85, 82, 80, 87, 90]
        // }
    };
    return (
        <Fragment>
            <Header/>
            <div className="soilMoistureDetailsSection">
                <div className="predictiveBarContent">
                        <h3 className="predictiveBarHeading">Predictive Analysis (AI Powered)</h3>
                        <div className="predictiveBarGraph">
                            <MyBarChart content = {apiData}/>
                        </div>
                </div>
                <div className="responsibilityChartContent">
                <h3 className="realTimeChartHeading">Real Time Chart (Moisture)</h3>
                    <div className="realTimePieChart">
                        <MyPieChart content = {apiData}/>
                    </div>
                </div>
            </div>
        </Fragment>
    );
};

export default SoilMoistureDetails;
