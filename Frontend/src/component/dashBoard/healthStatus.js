import React, { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import "./healthStatus.css";
import attention from "../../images/Attention.png";
import atRisk from "../../images/AtRisk.png";
import healthyLiving from "../../images/healthy-living.png";
import potatoPlant from "../../images/potatoPlant.png";
import CornPlant from "../../images/CornPlant.png";
import ricePlant from "../../images/Riceplant.png";

const HealthStatus = () => {
    const navigate = useNavigate();
    const { t } = useTranslation(); // Access translations

    const healthStatusData = [
        {
            mainImage : ricePlant,
            title : "Healthy",
            iconImage : healthyLiving
        },
        {
            mainImage : CornPlant,
            title : "Need Attention",
            iconImage : attention
        },
        {
            mainImage : potatoPlant,
            title : "At Risk",
            iconImage : atRisk
        }
    ]

    return (
        <div className="healthStatusSection">
            <div className="healthHeading">
                <h3>Crop Health Status</h3>
                <div className="allCropsBtn">All Crops</div>
            </div>
            <div className="healthContentBox">
                {healthStatusData.map((data)=>(
                    <div key={data.title} className="healthContentWrapper">
                        <img className="mainImage" src={data.mainImage} alt={data.title}/>
                        <h5 className="healthTitle">{data.title}</h5>
                        <img className="imageIcon" src={data.iconImage} alt={data.title}/>

                    </div>
                ))}
            </div>
        </div>
    );
};

export default HealthStatus;
