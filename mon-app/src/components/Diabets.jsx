import React, { useState } from 'react';
import imageDiabets from '../assets/images/diabetes.jpg';
import '../styles/Diabets.css';
import NavBar from '../pages/NavBar';

function Diabets() {
  const [inputdata, setInputdata] = useState({
    pregnancies: '',
    glucose: '',
    bloodPressure: '',
    skinThickness: '',
    insulin: '',
    bmi: '',
    diabetesPedigreeFunction: '',
    age: ''
  });

  const [resultat, setResultat] = useState('');
  const url = "http://localhost:8000/predictdiabete";

  // Fonction pour mettre à jour les valeurs des inputs
  const handleChange = (e) => {
    setInputdata({
      ...inputdata,
      [e.target.name]: e.target.value
    });
  };

  // Fonction pour envoyer les données
  const predictionD = (e) => {
    e.preventDefault(); // éviter le rechargement de la page

    fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(inputdata)
    })
      .then(res => res.json())
      .then(data => setResultat(data.result || data))
      .catch(err => console.error("Erreur prédiction :", err));
  };

  return (
    <>
      <NavBar />
      <div className="diabets-container">
        <h1>Diabetes disease Prediction</h1>
        <form className="form-card" onSubmit={predictionD}>
          <img src={imageDiabets} alt="diabetes" className="diabets-image" />

          <div className="grid-container">
            <div className="grid-item">
              <label htmlFor="pregnancies">Pregnancies:</label>
              <input type="number" id="pregnancies" name="pregnancies" value={inputdata.pregnancies} onChange={handleChange} required />
            </div>
            <div className="grid-item">
              <label htmlFor="glucose">Glucose:</label>
              <input type="number" id="glucose" name="glucose" value={inputdata.glucose} onChange={handleChange} required />
            </div>
            <div className="grid-item">
              <label htmlFor="bloodPressure">Blood Pressure:</label>
              <input type="number" id="bloodPressure" name="bloodPressure" value={inputdata.bloodPressure} onChange={handleChange} required />
            </div>
            <div className="grid-item">
              <label htmlFor="skinThickness">Skin Thickness:</label>
              <input type="number" id="skinThickness" name="skinThickness" value={inputdata.skinThickness} onChange={handleChange} required />
            </div>
            <div className="grid-item">
              <label htmlFor="insulin">Insulin:</label>
              <input type="number" id="insulin" name="insulin" value={inputdata.insulin} onChange={handleChange} required />
            </div>
            <div className="grid-item">
              <label htmlFor="bmi">BMI:</label>
              <input type="number" step="0.1" id="bmi" name="bmi" value={inputdata.bmi} onChange={handleChange} required />
            </div>
            <div className="grid-item">
              <label htmlFor="diabetesPedigreeFunction">Diabetes Pedigree Function:</label>
              <input type="number" step="0.01" id="diabetesPedigreeFunction" name="diabetesPedigreeFunction" value={inputdata.diabetesPedigreeFunction} onChange={handleChange} required />
            </div>
            <div className="grid-item">
              <label htmlFor="age">Age:</label>
              <input type="number" id="age" name="age" value={inputdata.age} onChange={handleChange} required />
            </div>
          </div>

          <button type="submit">Predict</button>
        </form>

        {resultat && (
          <p className="resultat">Résultat : {resultat}</p>
        )}
      </div>
    </>
  );
}

export default Diabets;
