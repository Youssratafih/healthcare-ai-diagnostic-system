import React from 'react';
import '../styles/Home.css';
import { Link } from 'react-router-dom';
import imageDiabets from '../assets/images/diabetes.jpg';
import imageTumor from '../assets/images/brain_tumor.jpg';
export default function Home() {
    const diseases = [
        {
          title: "Brain tumor",
          description: "Brain tumor is the most common cancer among women in the world. It accounts for 25% of all cancer cases.",
          image: imageTumor, // ✅ sans ../assets
          link: "/brain-tumor"
        },
        {
          title: "Diabetes",
          description: "Diabetes is a chronic disease that occurs when the body cannot properly use and store glucose and sugar.",
          image: imageDiabets, // ✅ sans ../assets
          link: "/diabetes"
        },
      ];
      
  


  return (
    <div className="home-container">
        <h1 className="health-title">Health IA</h1>

      <div className="cards-grid">
        {diseases.map((disease, index) => (
          <div className="card" key={index}>
            <img src={disease.image} alt={disease.title} />
            <h3>{disease.title}</h3>
            <p>{disease.description}</p>
            <Link to={disease.link} className="predict-button">Predict</Link>
          </div>
        ))}
      </div>
    </div>
  );
}
