export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4">
          <h1 className="text-3xl font-bold text-gray-900">
            OpsPilot – AI Assistant for IT Operations
          </h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="border-4 border-dashed border-gray-200 rounded-lg p-8">
            <h2 className="text-2xl font-semibold mb-4">Welcome to OpsPilot</h2>
            <p className="text-gray-600 mb-4">
              An AI-powered platform for IT operations automation, incident classification, 
              and intelligent remediation recommendations.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="font-semibold text-lg mb-2">📊 Analyze Logs</h3>
                <p className="text-gray-600 text-sm">
                  AI-powered log analysis to detect anomalies and patterns
                </p>
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="font-semibold text-lg mb-2">🎯 Classify Incidents</h3>
                <p className="text-gray-600 text-sm">
                  Automatic categorization and severity assessment
                </p>
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="font-semibold text-lg mb-2">🔧 Get Recommendations</h3>
                <p className="text-gray-600 text-sm">
                  Step-by-step remediation plans with verification
                </p>
              </div>
            </div>

            <div className="mt-8 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold mb-2">🚀 Quick Start</h3>
              <p className="text-sm text-gray-700">
                API is running at <code className="bg-gray-200 px-2 py-1 rounded">http://localhost:3001</code>
              </p>
              <p className="text-sm text-gray-700 mt-2">
                Try: <code className="bg-gray-200 px-2 py-1 rounded">POST /api/analyze-logs</code>
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
