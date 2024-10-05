import React from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import MetaData from "../metaData/metaData";
import "./getStarted.css";
import logo from "../../images/logo.png"; // Import your logo image

const GetStarted = () => {
    const navigate = useNavigate();
    const { t } = useTranslation(); // Access translations

  return (
    <div className="get_started">
      <MetaData title={t("get_started.title")} />

      {/* Logo at the top */}
      <div className="logo-container">
        <img src={logo} alt="App Logo" className="logo" />
      </div>

      {/* Get Started button */}
      <div className="content-container">
        <button onClick ={ ()=>{navigate("/login")}} className="get-started-button">
          {t("get_started.button")}
        </button>

        {/* Sign-up link */}
        <div className="sign-up-container">
          <p>
            {t("get_started.no_account")}{" "}
            <a href="/sign-up" className="sign-up-link">
              {t("get_started.sign_up")}
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default GetStarted;
