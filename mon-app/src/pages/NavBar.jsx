import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/NavBar.css';

export default function NavBar() {
  return (
    <nav className="custom-navbar">
      <ul className="nav-links">
        <li><Link to="/">Main</Link></li>
    
        <li><Link to="/diabetes">Diabetes</Link></li>
        
        <li><Link to="/brain-tumor">Brain Tumor</Link></li>
     
      </ul>

      <div className="hero-text">
        <h1>HealthAi intelligent disease Detection</h1>
        <p>Get predictions for various diseases using AI models.</p>
      </div>
    </nav>
  );
}
