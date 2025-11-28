import React, { useState } from 'react';
import CropRecommendation from './components/CropRecommendation';
import FertilizerOptimization from './components/FertilizerOptimization';
import PestDetection from './components/PestDetection';
import { LeafIcon, DropletIcon, BugIcon } from './components/icons';

type AITool = 'crop' | 'fertilizer' | 'pest';

const App: React.FC = () => {
  const [activeTool, setActiveTool] = useState<AITool>('pest');

  const renderTool = () => {
    switch (activeTool) {
      case 'crop':
        return <CropRecommendation />;
      case 'fertilizer':
        return <FertilizerOptimization />;
      case 'pest':
        return <PestDetection />;
      default:
        return <PestDetection />;
    }
  };

  const NavButton: React.FC<{
    tool: AITool;
    label: string;
    children: React.ReactNode;
  }> = ({ tool, label, children }) => (
    <button
      onClick={() => setActiveTool(tool)}
      className={`flex items-center justify-center w-full sm:w-auto px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-white/50 ${
        activeTool === tool
          ? 'bg-white text-slate-800 shadow-lg'
          : 'text-white/80 hover:bg-white/20 hover:text-white'
      }`}
      aria-selected={activeTool === tool}
      role="tab"
    >
      {children}
      <span className="ml-2">{label}</span>
    </button>
  );

  return (
    <div className="bg-black min-h-screen text-slate-800 p-4 sm:p-6 lg:p-8">
      <div className="max-w-5xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl sm:text-5xl font-bold text-white tracking-tight">Agri-AI Suite</h1>
          <p className="mt-3 text-lg text-slate-300">Your Smart Farming Assistant</p>
        </header>

        <nav className="flex flex-col sm:flex-row gap-2 p-1.5 bg-gray-900/70 backdrop-blur-sm rounded-lg sticky top-4 z-10 border border-white/20" role="tablist" aria-label="AI Tools">
          <NavButton tool="crop" label="Crop Recommendation">
            <LeafIcon className="w-5 h-5" />
          </NavButton>
          <NavButton tool="fertilizer" label="Fertilizer Optimization">
            <DropletIcon className="w-5 h-5" />
          </NavButton>
          <NavButton tool="pest" label="Pest & Disease Detection">
            <BugIcon className="w-5 h-5" />
          </NavButton>
        </nav>

        <main className="mt-8">
          {renderTool()}
        </main>
      </div>
    </div>
  );
};

export default App;