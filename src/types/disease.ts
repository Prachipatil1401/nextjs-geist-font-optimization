export interface Disease {
  id: string;
  name: string;
  description: string;
  symptoms: string[];
  treatments: Treatment[];
  severity: 'low' | 'medium' | 'high';
  confidence?: number;
}

export interface Treatment {
  id: string;
  name: string;
  type: 'organic' | 'chemical' | 'biological';
  description: string;
  applicationMethod: string;
  dosage: string;
  timing: string;
  precautions: string[];
  effectiveness: number; // 0-100
}

export interface DetectionResult {
  id: string;
  imageUrl: string;
  detectedDiseases: Disease[];
  isHealthy: boolean;
  confidence: number;
  timestamp: Date;
  location?: {
    latitude: number;
    longitude: number;
  };
  notes?: string;
}

export interface AnalysisHistory {
  id: string;
  results: DetectionResult[];
  totalAnalyses: number;
  healthyCount: number;
  diseasedCount: number;
}
