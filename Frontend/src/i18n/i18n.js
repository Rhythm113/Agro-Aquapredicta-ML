import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import LanguageDetector from "i18next-browser-languagedetector";

// Import the JSON files
import enTranslation from './language/en.json';
import bnTranslation from './language/bn.json';
// import frTranslation from './language/fr.json';
// import deTranslation from './language/de.json';

// Add as many languages as you need
const resources = {
  en: { translation: enTranslation },
  bn: { translation: bnTranslation },
//   fr: { translation: frTranslation },
//   de: { translation: deTranslation }
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: "en", // Default language
    interpolation: {
      escapeValue: false, // React already escapes by default
    }
  });

export default i18n;
