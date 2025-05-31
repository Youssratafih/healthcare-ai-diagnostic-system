import { Routes, Route, BrowserRouter } from "react-router-dom"; // 👈 Utiliser react-router-dom
import Diabets from "./components/Diabets";
import Brain_Tumor from "./components/Brain_Tumor";
import Home from "./components/Home";

function App() {
  return (
    <>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/diabetes" element={<Diabets />} />
        <Route path="/brain-tumor" element={<Brain_Tumor />} />
      </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;
