import React, { useState, useEffect } from "react";
import WebFont from "webfontloader";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import SelectLanguage from "./component/starter/starterLanguage";
import GetStarted from "./component/starter/getStarted";
import Login from "./component/logInregister/login";
import DashBoard from "./component/dashBoard/dashBoard";
import InitialLoader from "./component/loader/initialLoader";
import SoilMoistureDetails from "./component/soilMoisture/soilMoisture";

function App() {
  const [appLoading, setAppLoading] = useState(true);

  useEffect(() => {
    WebFont.load({
      google: {
        families: ["Roboto", "Droid Sans", "Chilanka"],
      },
    });

    // Simulate a loading delay for the app loading state
    const timer = setTimeout(() => {
      setAppLoading(false);
    }, 3000);

    // Cleanup timeout on component unmount
    return () => clearTimeout(timer);
  }, []);

  if (appLoading) {
    return <InitialLoader />; // Show a loader until the app is ready
  }

  return (
    <Router>
      <Routes>
        <Route exact path="/" Component={SelectLanguage} />
        <Route exact path="/getStarted" Component={GetStarted} />
        <Route exact path="/login" Component={Login} />
        <Route exact path="/dashboard" Component={DashBoard} />
        <Route exact path="/soil" Component={SoilMoistureDetails} />
      </Routes>
    </Router>
  );
}

export default App;
