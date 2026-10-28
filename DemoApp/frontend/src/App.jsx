import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainMenu from './pages/MainMenu';
import ClassifierV13 from './pages/ClassifierV13';
import ClassifierV14 from './pages/ClassifierV14';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainMenu />} />
        <Route path="/v13" element={<ClassifierV13 />} />
        <Route path="/v14" element={<ClassifierV14 />} />
      </Routes>
    </Router>
  );
}

export default App;
