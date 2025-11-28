import React from 'react';

interface ResultCardProps {
    title: string;
    content: string;
    icon: React.ReactNode;
}

const ResultCard: React.FC<ResultCardProps> = ({ title, content, icon }) => {
    // Simple markdown to HTML conversion with updated styling
    const formatContent = (text: string) => {
        return text
            .split('\n')
            .map(line => {
                if (line.startsWith('### ')) return `<h3 class="text-md font-semibold mt-4 mb-1 text-slate-800">${line.substring(4)}</h3>`;
                if (line.startsWith('## ')) return `<h2 class="text-lg font-bold mt-4 mb-2 text-slate-800">${line.substring(3)}</h2>`;
                if (line.startsWith('# ')) return `<h1 class="text-xl font-bold mt-6 mb-3 text-slate-900">${line.substring(2)}</h1>`;
                if (line.startsWith('* ')) return `<li class="ml-5 list-disc text-slate-600">${line.substring(2)}</li>`;
                if (line.trim() === '') return '<br />';
                return `<p class="text-slate-600">${line}</p>`;
            })
            .join('');
    };

    return (
        <div className="mt-8 md:mt-0 border-t border-slate-200 pt-6 md:border-none md:pt-0">
            <div className="flex items-center mb-4">
                {icon}
                <h2 className="text-xl font-semibold text-slate-800">{title}</h2>
            </div>
            <div
                className="prose max-w-none prose-p:text-slate-600 prose-headings:text-slate-800"
                dangerouslySetInnerHTML={{ __html: formatContent(content) }}
            />
        </div>
    );
};

export default ResultCard;