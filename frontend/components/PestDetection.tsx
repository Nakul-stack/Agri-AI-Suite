import React, { useState, useRef } from 'react';
import { detectPestOrDisease } from '../services/geminiService';
import { fileToBase64 } from '../utils/fileUtils';
import { BugIcon, ResetIcon, UploadIcon, AwaitingAnalysisIcon } from './icons';
import Spinner from './Spinner';
import ResultCard from './ResultCard';

const PestDetection: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreviewUrl(URL.createObjectURL(selectedFile));
            setResult(null);
            setError(null);
        }
    };

    const handleReset = () => {
        setFile(null);
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
        }
        setPreviewUrl(null);
        setResult(null);
        setError(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleSubmit = async () => {
        if (!file) {
            setError('Please upload an image first.');
            return;
        }
        setLoading(true);
        setError(null);
        setResult(null);
        try {
            const imageB64 = await fileToBase64(file);
            const detectionResult = await detectPestOrDisease(imageB64, file.type);
            setResult(detectionResult);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white p-6 sm:p-8 rounded-xl shadow-sm border border-slate-200/80 fade-in">
            <div className="flex items-center mb-6">
                 <div className="p-2 bg-slate-100 rounded-full mr-3">
                    <BugIcon className="w-6 h-6 text-slate-600" />
                </div>
                <h2 className="text-xl font-semibold text-slate-800">Upload Plant Image for Analysis</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="flex-1">
                    <input
                        type="file"
                        accept="image/png, image/jpeg"
                        onChange={handleFileChange}
                        className="hidden"
                        ref={fileInputRef}
                    />
                    <div
                        className="border-2 border-dashed border-slate-300 rounded-lg p-6 text-center cursor-pointer hover:border-slate-400 hover:bg-slate-50/50 transition-colors flex items-center justify-center min-h-[260px]"
                        onClick={() => fileInputRef.current?.click()}
                    >
                        {previewUrl ? (
                            <img src={previewUrl} alt="Preview" className="mx-auto max-h-60 w-full object-contain rounded-md" />
                        ) : (
                            <div className="flex flex-col items-center justify-center">
                                <UploadIcon className="w-10 h-10 mx-auto text-slate-400" />
                                <p className="mt-2 text-sm text-slate-600">
                                    <span className="font-semibold text-slate-700">Click to upload</span> or drag and drop
                                </p>
                                <p className="text-xs text-slate-500">PNG, JPG, up to 10MB</p>
                            </div>
                        )}
                    </div>

                    <div className="mt-6 flex flex-col sm:flex-row items-center gap-4">
                        <button
                            onClick={handleSubmit}
                            disabled={!file || loading}
                            className="w-full sm:w-auto flex items-center justify-center px-6 py-2.5 bg-slate-300 text-slate-800 font-semibold rounded-lg shadow-sm hover:bg-slate-400/80 disabled:bg-slate-200 disabled:text-slate-500 disabled:cursor-not-allowed transition-colors duration-200"
                        >
                            <BugIcon className="w-5 h-5 mr-2" />
                            {loading ? 'Analyzing...' : 'Analyze Image'}
                        </button>
                        <button
                            type="button"
                            onClick={handleReset}
                            className="w-full sm:w-auto flex items-center justify-center px-6 py-2.5 bg-white text-slate-700 border border-slate-300 font-semibold rounded-lg shadow-sm hover:bg-slate-100 disabled:opacity-50 transition-colors duration-200"
                        >
                            <ResetIcon className="w-5 h-5 mr-2" />
                            Clear
                        </button>
                    </div>
                </div>
                
                <div className="flex-1 md:border-l md:pl-8 border-slate-200/80 flex items-center justify-center">
                    {loading && <Spinner />}
                    {error && <div className="p-4 bg-red-50 text-red-700 rounded-lg border border-red-200 w-full">{error}</div>}
                    {result && <ResultCard title="Analysis Report" content={result} icon={<BugIcon className="w-6 h-6 text-slate-600 mr-3" />} />}
                    {!loading && !result && !error && (
                        <div className="p-6 bg-slate-100 rounded-lg h-full w-full flex flex-col justify-center items-center text-center">
                            <AwaitingAnalysisIcon className="w-16 h-16 text-slate-400 mb-4" />
                            <h3 className="text-lg font-semibold text-slate-700">Awaiting Image Analysis</h3>
                            <p className="text-slate-500 mt-1 text-sm max-w-xs">Upload an image and click "Analyze Image" to see the AI-powered report here.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default PestDetection;