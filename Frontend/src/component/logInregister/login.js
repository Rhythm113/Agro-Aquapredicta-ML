import React, {Fragment, useState} from "react";
import { useTranslation } from "react-i18next";
import MetaData from "../metaData/metaData";
import "./login.css";
import logo from "../../images/logo.png";
import { FaEye, FaEyeSlash } from "react-icons/fa"; 
import { useNavigate } from "react-router-dom";
import Loader from "../loader/loader";

const Login = () => {
  const { t } = useTranslation();
  const navigate = useNavigate()
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showError, setError] = useState("");
  const [loading, setLoading] = useState(false);


  const loginSubmit =async (e) => {
    e.preventDefault();

    try {
      setLoading(true); // Set loading to true while processing
      
    //   login dispatch here
      console.log("Successfully Logged In");
      localStorage.setItem("user", {email, password})
      navigate("/dashboard"); 
    } catch (error) {
      console.error("Error Log In:", error);
      setError(error) // Handle any errors during deletion
    } finally {
      setLoading(false); // Set loading to false after deletion completes
    }

    
  };

  return (
    <Fragment>
      <MetaData title={t("login.title")} />
      {loading ? (<Loader/>) :(
        <Fragment>
             <div className="login">

                <div className="logo-container">
                    <img src={logo} alt="App Logo" className="logo" />
                </div>

                <div className="login-container">
                    <h3 className="loginTitle">{t("login.title")}</h3>
                    <div className="email_passBox">
                        <input
                            type="email"
                            placeholder={t("login.email_placeholder")}
                            required
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                        />
                        <div className="passwordWrapper">
                            <input
                                type={showPassword ? "text" : "password"}
                                id="password"
                                placeholder={t("login.pass_placeholder")}
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                            <span
                                onClick={() => setShowPassword(!showPassword)}
                                className="eyeIcon"
                            >
                                {showPassword ? <FaEyeSlash /> : <FaEye />}
                            </span>
                        </div>
                    </div>

                    <div className="forgotPassBox">
                        <p onClick={()=>{navigate("/forgot_pass")}}>{t("login.forgot_pass")}</p>
                    </div>

                    <button onClick = {loginSubmit} className="login-button">
                        {t("login.button")}
                    </button>

                </div>
            </div>
        </Fragment>
      )}
    </Fragment>
   
  );
};

export default Login;
