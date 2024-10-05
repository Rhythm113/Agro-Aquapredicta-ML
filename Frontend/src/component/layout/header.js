import React from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import "./header.css";
import logoIcon from "../../images/Only Logo.png";
import logoText from "../../images/Chashi Sheba.png";
import profileIcon from "../../images/account.png";
import locationIcon from "../../images/location.png";

const Header = () => {
    const navigate = useNavigate();
    const { t } = useTranslation(); // Access translations

  return (
    <div className="headerSection"> 
        <div className="logoBox">
            <div className="logoBoxWrapper">
                <img src={logoIcon} alt="Logo Icon"/>
                <img src={logoText} alt="Logo Text"/>
            </div>
        </div>
        <div className="headerIconBox">
            <img src={profileIcon} alt="Profile Icon"/>
            <img src={locationIcon} alt="Location Icon"/>
        </div>
    </div>
  );
};

export default Header;
