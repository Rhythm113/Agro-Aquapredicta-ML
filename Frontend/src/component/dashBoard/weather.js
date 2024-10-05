import React, { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import "./weather.css";
import sunny from "../../images/Sunny.gif";
import windSunny from "../../images/Windy+Sunny.gif";
import rainy from "../../images/Rainy.gif";
import cloudy from "../../images/Cloudy.gif";
import temperature from "../../images/temperature.png";
import precipitation from "../../images/precipitation.png";
import humidity from "../../images/humidity.png";
import wind from "../../images/wind-turbine (1).png";
import sunIcon from "../../images/sun.png";

const Weather = () => {
    const navigate = useNavigate();
    const { t } = useTranslation(); // Access translations

    const weatherType = [
        { type: "sunny", imgSrc: sunny },
        { type: "windSunny", imgSrc: windSunny },
        { type: "rainy", imgSrc: rainy },
        { type: "cloudy", imgSrc: cloudy },
    ];

    const weatherComponent = [
        { title: "Soil temp", imgSrc: temperature, value: "22°C" },
        { title: "Humidity", imgSrc: humidity, value: "59%" },
        { title: "Wind", imgSrc: wind, value: "6m/s" },
        { title: "Precipitation", imgSrc: precipitation, value: "0mm" },
    ];

    const currentWeatherType = "cloudy"; // Current weather condition

    const sunriseTime = 6; // 6 AM
    const sunsetTime = 18; // 6 PM

    // Sun position state
    const [sunPosition, setSunPosition] = useState({ x: 150, y: 40 });
    const pathRef = useRef(null); // Ref for the SVG path

    useEffect(() => {
        const calculateSunPosition = () => {
            const currentTime = new Date().getHours(); // Get current hour (0-23)
            
            if (currentTime >= sunriseTime && currentTime <= sunsetTime && pathRef.current) {
                const totalDaylightHours = sunsetTime - sunriseTime;
                const hoursSinceSunrise = currentTime - sunriseTime;

                // Calculate percentage of the day passed
                const dayProgress = hoursSinceSunrise / totalDaylightHours;

                // Get the total length of the SVG path
                const pathLength = pathRef.current.getTotalLength();

                // Get the point at the length corresponding to the day progress
                const point = pathRef.current.getPointAtLength(pathLength * dayProgress);

                // Set the new sun position based on the point
                setSunPosition({ x: point.x, y: point.y });
            } else if (currentTime < sunriseTime) {
                // Before sunrise
                const point = pathRef.current.getPointAtLength(0); // Start of the curve
                setSunPosition({ x: point.x, y: point.y });
            } else {
                // After sunset
                const pathLength = pathRef.current.getTotalLength();
                const point = pathRef.current.getPointAtLength(pathLength); // End of the curve
                setSunPosition({ x: point.x, y: point.y });
            }
        };

        calculateSunPosition(); // Initial position calculation
        const interval = setInterval(calculateSunPosition, 60000); // Update every minute

        return () => clearInterval(interval); // Cleanup interval on unmount
    }, [sunriseTime, sunsetTime]);

    return (
        <div className="weatherSection">
            <div className="weatherLeft">
                <div className="addressDateAndTemp">
                    <div className="addressAndDate">
                        <h5>Pahartoli, Raozan</h5>
                        <p>Tuesday, 1st October 2024</p>
                    </div>
                </div>
                <div className="Temp">
                    <h3>23°C</h3>
                </div>

                <div className="weatherComponent">
                    {weatherComponent.map((comp) => (
                        <div key={comp.title} className="weatherCompDetails">
                            <img src={comp.imgSrc} alt={comp.title} />
                            <p>{comp.value}<br />{comp.title}</p>
                        </div>
                    ))}
                </div>

                <div className="sunPosition">
                    {/* SVG for sunrise, sunset, and moving sun */}
                    <svg width="300" height="75" viewBox="0 20 300 100">
                        {/* Dotted curve line */}
                        <path
                            ref={pathRef}
                            d="M 30,80 Q 150,10 270,80"
                            stroke="black"
                            strokeWidth="2"
                            strokeDasharray="5,5"
                            fill="transparent"
                        />
                        {/* Sunrise point */}
                        <circle cx="30" cy="80" r="5" fill="orange" />
                        <text x="30" y="100" fontSize="12" textAnchor="middle">Sunrise</text>
                        {/* Sunset point */}
                        <circle cx="270" cy="80" r="5" fill="orange" />
                        <text x="270" y="100" fontSize="12" textAnchor="middle">Sunset</text>
                        {/* Moving sun */}
                        <image href={sunIcon} x={sunPosition.x - 15} y={sunPosition.y - 15} height="30px" width="30px" />
                    </svg>
                </div>
            </div>

            <div className="weatherRight">
                {weatherType
                    .filter((weather) => weather.type === currentWeatherType)
                    .map((weather) => (
                        <img key={weather.type} src={weather.imgSrc} alt={weather.type} />
                    ))}
            </div>
        </div>
    );
};

export default Weather;
