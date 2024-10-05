import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import MetaData from "../metaData/metaData";
import logo from "../../images/logo.png"
import "./starterLanguage.css";

const SelectLanguage = () => {
  const { t, i18n } = useTranslation(); 
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState("");
  const navigate = useNavigate();

  // useEffect(() => {
  //   // Check if user data exists in local storage
  //   const userData = localStorage.getItem("user");
  //   if (userData) {
  //     // If user data exists, navigate to dashboard
  //     navigate("/dashboard");
  //   }
  // }, [navigate]);

  const languages = [
    { code: "en", label: "English" },
    { code: "bn", label: "Bengali" },
    // { code: "fr", label: "French" },
    // { code: "de", label: "German" },
  ];

  const filteredLanguages = languages.filter((language) =>
    language.label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleLanguageSelect = (language) => {
    setSelectedLanguage(language);
    i18n.changeLanguage(language.code); // Change language
  };

  const handleNext = () => {
    if (selectedLanguage) {
      console.log(`Selected Language: ${selectedLanguage.label}`);
      navigate("/getStarted");
    } else {
      alert(t("select_language"));
    }
  };

  return (
    <div className="language-container">
      <MetaData title="Select Language" />
      <div className="logoContetnInLanguage">
        <img src={logo} alt="App Logo" className="logo" />
      </div>
      <div>
        <div className="language-header">
          <h1>Select your language</h1>
          <input
            type="text"
            className="search-bar"
            placeholder="Search your language"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="language-list">
          {filteredLanguages.length > 0 ? (
            filteredLanguages.map((language) => (
              <div
                key={language.code}
                className={`language-item ${
                  selectedLanguage.label === language.label ? "selected" : ""
                }`}
                onClick={() => handleLanguageSelect(language)}
              >
                {language.label}
              </div>
            ))
          ) : (
            <div className="no-results">No language found</div>
          )}
        </div>
      </div>
      {selectedLanguage && (
        <button className="next-button" onClick={handleNext}>
           {t("next")} &#x2192;
        </button>
      )}
    </div>
  );
};

export default SelectLanguage;
