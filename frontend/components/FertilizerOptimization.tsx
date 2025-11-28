import React, { useState } from 'react';
import { getFertilizerRecommendation } from '../services/geminiService';
import type { FertilizerData } from '../types';
import { SOIL_TYPES, CROP_TYPES } from '../constants';
import { DropletIcon, ResetIcon, MoleculeIcon, TemperatureIcon, HumidityIcon, PhIcon, SoilMoistureIcon, SoilTypeIcon, CropTypeIcon } from './icons';
import Spinner from './Spinner';
import ResultCard from './ResultCard';

const initialFormData: FertilizerData = {
    nitrogen: '',
    phosphorus: '',
    potassium: '',
    temperature: '',
    humidity: '',
    soilMoisture: '',
    soilType: SOIL_TYPES[0],
    cropType: CROP_TYPES[0],
    ph: '',
};

const FertilizerOptimization: React.FC = () => {
    const [formData, setFormData] = useState<FertilizerData>(initialFormData);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<string | null>(null);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleReset = () => {
        setFormData(initialFormData);
        setResult(null);
        setError(null);
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);
        try {
            const recommendation = await getFertilizerRecommendation(formData);
            setResult(recommendation);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setLoading(false);
        }
    };
    
    const FormInput = ({ name, label, unit, icon, placeholder, required = true }: { name: keyof Omit<FertilizerData, 'soilType'|'cropType'>, label: string, unit: string, icon: React.ReactNode, placeholder: string, required?: boolean }) => (
        <div>
            <label htmlFor={name} className="flex items-center text-sm font-medium text-slate-600 mb-1">
                {icon}
                <span className="ml-2">{label}</span>
            </label>
            <input
                type="number"
                id={name}
                name={name}
                value={formData[name]}
                onChange={handleChange}
                className="w-full p-2 bg-white border border-slate-300 rounded-md shadow-sm focus:ring-1 focus:ring-slate-500 focus:border-slate-500 transition"
                placeholder={placeholder}
                required={required}
            />
            <p className="text-xs text-slate-500 mt-1">{unit}</p>
        </div>
    );
    
    const FormSelect = ({ name, label, options, icon }: { name: keyof FertilizerData, label: string, options: readonly string[], icon: React.ReactNode }) => (
        <div>
            <label htmlFor={name} className="flex items-center text-sm font-medium text-slate-600 mb-1">
                {icon}
                <span className="ml-2">{label}</span>
            </label>
            <select
                id={name}
                name={name}
                value={formData[name]}
                onChange={handleChange}
                className="w-full p-2 bg-white border border-slate-300 rounded-md shadow-sm focus:ring-1 focus:ring-slate-500 focus:border-slate-500 transition"
            >
                {options.map(option => <option key={option} value={option}>{option}</option>)}
            </select>
        </div>
    );


    return (
        <div className="bg-white p-6 sm:p-8 rounded-xl shadow-sm border border-slate-200/80 fade-in">
            <div className="flex items-center mb-6">
                <div className="p-2 bg-slate-100 rounded-full mr-3">
                    <DropletIcon className="w-6 h-6 text-slate-600" />
                </div>
                <h2 className="text-xl font-semibold text-slate-800">Enter Soil Data for Fertilizer</h2>
            </div>
            <form onSubmit={handleSubmit}>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                    <FormInput name="nitrogen" label="Nitrogen (N)" unit="kg/ha" icon={<MoleculeIcon className="w-4 h-4 text-slate-500" />} placeholder="e.g., 100" />
                    <FormInput name="phosphorus" label="Phosphorus (P)" unit="kg/ha" icon={<MoleculeIcon className="w-4 h-4 text-slate-500" />} placeholder="e.g., 18" />
                    <FormInput name="potassium" label="Potassium (K)" unit="kg/ha" icon={<MoleculeIcon className="w-4 h-4 text-slate-500" />} placeholder="e.g., 20" />
                    <FormInput name="temperature" label="Temperature" unit="°C" icon={<TemperatureIcon className="w-4 h-4 text-slate-500" />} placeholder="e.g., 29" />
                    <FormInput name="humidity" label="Humidity" unit="%" icon={<HumidityIcon className="w-4 h-4 text-slate-500" />} placeholder="e.g., 94" />
                    <FormInput name="soilMoisture" label="Soil Moisture" unit="%" icon={<SoilMoistureIcon className="w-4 h-4 text-slate-500" />} placeholder="e.g., 40" />
                    <FormSelect name="soilType" label="Soil Type" options={SOIL_TYPES} icon={<SoilTypeIcon className="w-4 h-4 text-slate-500" />} />
                    <FormSelect name="cropType" label="Crop Type" options={CROP_TYPES} icon={<CropTypeIcon className="w-4 h-4 text-slate-500" />} />
                    <FormInput name="ph" label="Soil pH (Optional)" unit="pH scale" icon={<PhIcon className="w-4 h-4 text-slate-500" />} placeholder="e.g., 7.5" required={false} />
                </div>
                 <div className="mt-8 flex flex-col sm:flex-row items-center gap-4">
                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full sm:w-auto flex items-center justify-center px-6 py-2.5 bg-slate-800 text-white font-semibold rounded-lg shadow-sm hover:bg-slate-700 disabled:bg-slate-400 disabled:cursor-not-allowed transition-colors duration-200"
                    >
                        <DropletIcon className="w-5 h-5 mr-2" />
                        {loading ? 'Analyzing...' : 'Get Recommendation'}
                    </button>
                    <button
                        type="button"
                        onClick={handleReset}
                        className="w-full sm:w-auto flex items-center justify-center px-6 py-2.5 bg-white text-slate-700 border border-slate-300 font-semibold rounded-lg shadow-sm hover:bg-slate-100 disabled:opacity-50 transition-colors duration-200"
                    >
                        <ResetIcon className="w-5 h-5 mr-2" />
                        Reset
                    </button>
                </div>
            </form>
            {loading && <div className="mt-8"><Spinner /></div>}
            {error && <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-lg border border-red-200">{error}</div>}
            {result && <div className="mt-8"><ResultCard title="Fertilizer Recommendation" content={result} icon={<DropletIcon className="w-6 h-6 text-slate-600 mr-3" />} /></div>}
        </div>
    );
};

export default FertilizerOptimization;