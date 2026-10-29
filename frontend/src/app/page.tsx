"use client";

import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import Latex from "react-latex-next";
import {
  CloudUpload,
  FileText,
  ChevronUp,
  ChevronDown,
  Send,
  Sparkles,
  BookOpen,
  Lock,
  Unlock,
  HelpCircle,
  Volume2,
  Video,
  PlayCircle,
  BrainCircuit,
} from "lucide-react";

const API_BASE_URL = "http://127.0.0.1:8000";

interface LogicSandbox {
  question_markdown: string;
  starter_template: string;
  hint: string;
  grading_rubric: string;
}

interface ValidatedVideoItem {
  topic_title: string;
  query: string;
  video_id: string;
  embed_url: string;
  validation_summary: string;
}

interface YouTubeValidation {
  video_id: string;
  embed_url: string;
  validation_summary: string;
  validated_criteria: string[];
  playlist?: ValidatedVideoItem[];
}

interface RoutingPayload {
  classification: "LOGIC" | "SPATIAL" | "ABSTRACT";
  confidence: number;
  core_concepts: string[];
  youtube_search_queries: string[];
  transcript_validation_criteria: string[];
  spatial_explainer_prompt: string;
  abstract_podcast_script: string;
  youtube_validation?: YouTubeValidation;
  practice_sandbox?: LogicSandbox;
}

interface DocumentItem {
  file_hash: string;
  filename: string;
  status: "PROCESSING" | "EXTRACTED" | "ROUTED" | "GENERATING" | "COMPLETE" | "ERROR";
  progress: number;
  taxonomy_classification?: "LOGIC" | "SPATIAL" | "ABSTRACT" | null;
  routing_payload?: RoutingPayload | null;
  raw_extracted_text?: string | null;
  generated_media_path?: string | null;
  pdf_url?: string | null;
  knowledge_summary?: string | null;
  error_message?: string | null;
}

interface ChatMessage {
  id: string;
  sender: "ai" | "user";
  text: string;
  citations?: string[];
}

export default function NorthStarDualPaneApp() {
  // Sidebar & Upload State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploadCollapsed, setIsUploadCollapsed] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Documents & Dual-Pane View State
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [activeHash, setActiveHash] = useState<string | null>(null);
  const [leftPaneMode, setLeftPaneMode] = useState<"pdf" | "markdown">("pdf");

  // Multi-Video Playlist Tab State (keyed by file_hash)
  const [activeVideoTab, setActiveVideoTab] = useState<Record<string, number>>({});

  // Path A (LOGIC) Active Practice Sandbox State
  const [sandboxInput, setSandboxInput] = useState<Record<string, string>>({});
  const [sandboxUnlocked, setSandboxUnlocked] = useState<Record<string, boolean>>({});
  const [sandboxFeedback, setSandboxFeedback] = useState<Record<string, string>>({});
  const [isVerifyingSandbox, setIsVerifyingSandbox] = useState(false);
  const [showHint, setShowHint] = useState(false);

  // Phase 5 Socratic RAG Chat & Asymmetric Memory State
  const [chatHistories, setChatHistories] = useState<Record<string, ChatMessage[]>>({});
  const [knowledgeSummaries, setKnowledgeSummaries] = useState<Record<string, string>>({});
  const [chatInput, setChatInput] = useState("");
  const [isAiTyping, setIsAiTyping] = useState(false);

  const pollingRefs = useRef<Record<string, NodeJS.Timeout>>({});
  const activeDoc = documents.find((d) => d.file_hash === activeHash) || null;

  useEffect(() => {
    const currentPolls = pollingRefs.current;
    return () => {
      Object.values(currentPolls).forEach(clearInterval);
    };
  }, []);

  // Initialize starter template & Socratic greeting when a document completes
  useEffect(() => {
    if (!activeDoc) return;
    const hash = activeDoc.file_hash;

    if (
      activeDoc.routing_payload?.practice_sandbox?.starter_template &&
      sandboxInput[hash] === undefined
    ) {
      setSandboxInput((prev) => ({
        ...prev,
        [hash]: activeDoc.routing_payload!.practice_sandbox!.starter_template,
      }));
    }

    if (
      (activeDoc.status === "COMPLETE" || activeDoc.status === "ROUTED") &&
      !chatHistories[hash]
    ) {
      const firstConcept =
        activeDoc.routing_payload?.core_concepts?.[0] || "the core lecture material";
      setChatHistories((prev) => ({
        ...prev,
        [hash]: [
          {
            id: "init-socratic",
            sender: "ai",
            text: `Socratic RAG Tutor ready. Ask a question about ${firstConcept} or click "Challenge Me" below for an active-recall pop quiz grounded in your slides.`,
            citations: ["Local Vector Index Ready"],
          },
        ],
      }));
    }
  }, [activeDoc, chatHistories, sandboxInput]);

  // Phase 4: 5-Second Polling Loop until status === 'COMPLETE'
  const startPolling = (fileHash: string) => {
    if (pollingRefs.current[fileHash]) {
      clearInterval(pollingRefs.current[fileHash]);
    }

    pollingRefs.current[fileHash] = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/status/${fileHash}`);
        const data = res.data;

        if (data.knowledge_summary) {
          setKnowledgeSummaries((prev) => ({
            ...prev,
            [fileHash]: data.knowledge_summary,
          }));
        }

        setDocuments((prev) =>
          prev.map((doc) => {
            if (doc.file_hash !== fileHash) return doc;

            if (data.status === "COMPLETE") {
              clearInterval(pollingRefs.current[fileHash]);
              delete pollingRefs.current[fileHash];
              return {
                ...doc,
                status: "COMPLETE",
                progress: 100,
                taxonomy_classification: data.taxonomy_classification,
                routing_payload: data.routing_payload,
                raw_extracted_text: data.raw_extracted_text,
                generated_media_path: data.generated_media_path,
                pdf_url: data.pdf_url,
              };
            }

            const progressMap: Record<string, number> = {
              PROCESSING: 30,
              EXTRACTED: 55,
              ROUTED: 75,
              GENERATING: 90,
            };

            return {
              ...doc,
              status: data.status,
              progress: progressMap[data.status] || Math.min(doc.progress + 10, 92),
              taxonomy_classification: data.taxonomy_classification,
              routing_payload: data.routing_payload,
              raw_extracted_text: data.raw_extracted_text,
              pdf_url: data.pdf_url,
            };
          })
        );
      } catch (err: any) {
        if (err?.response?.status === 404) {
          clearInterval(pollingRefs.current[fileHash]);
          delete pollingRefs.current[fileHash];
          setDocuments((prev) =>
            prev.map((doc) =>
              doc.file_hash === fileHash
                ? {
                    ...doc,
                    status: "ERROR",
                    progress: 0,
                    error_message: "Worker failed or quota exceeded.",
                  }
                : doc
            )
          );
        }
      }
    }, 5000);
  };

  // Upload Handlers
  const handleUpload = async () => {
    if (!selectedFile) {
      fileInputRef.current?.click();
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    const uploadedFilename = selectedFile.name;

    try {
      const res = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const data = res.data;
      const isComplete = data.status === "COMPLETE";

      const newDoc: DocumentItem = {
        file_hash: data.file_hash,
        filename: uploadedFilename,
        status: data.status,
        progress: isComplete ? 100 : 25,
        taxonomy_classification: data.taxonomy_classification,
        routing_payload: data.routing_payload,
        raw_extracted_text: data.raw_extracted_text,
        generated_media_path: data.generated_media_path,
        pdf_url: data.pdf_url,
      };

      if (data.knowledge_summary) {
        setKnowledgeSummaries((prev) => ({
          ...prev,
          [data.file_hash]: data.knowledge_summary,
        }));
      }

      setDocuments((prev) => [
        newDoc,
        ...prev.filter((d) => d.file_hash !== data.file_hash),
      ]);
      setActiveHash(data.file_hash);
      setSelectedFile(null);
      setIsUploadCollapsed(true);

      if (!isComplete) {
        startPolling(data.file_hash);
      }
    } catch (err: any) {
      alert(
        err?.response?.data?.detail ||
          "Cannot reach FastAPI backend on http://127.0.0.1:8000"
      );
    } finally {
      setIsUploading(false);
    }
  };

  // Phase 4: Verify Path A (LOGIC) Sandbox Submission
  const handleVerifySandbox = async () => {
    if (!activeDoc) return;
    const hash = activeDoc.file_hash;
    const submission = sandboxInput[hash] || "";
    if (!submission.trim()) return;

    setIsVerifyingSandbox(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/verify-challenge`, {
        file_hash: hash,
        student_submission: submission,
      });
      setSandboxFeedback((prev) => ({ ...prev, [hash]: res.data.feedback }));
      if (res.data.passed) {
        setSandboxUnlocked((prev) => ({ ...prev, [hash]: true }));
      }
    } catch {
      setSandboxFeedback((prev) => ({
        ...prev,
        [hash]: "Error verifying solution with backend.",
      }));
    } finally {
      setIsVerifyingSandbox(false);
    }
  };

  // Phase 5: Socratic RAG Chat & "Challenge Me" Trigger
  const sendSocraticMessage = async (text: string, isChallengeMe: boolean = false) => {
    if (!activeDoc) return;
    const hash = activeDoc.file_hash;
    const displayUserMsg = isChallengeMe
      ? "⚡ Challenge Me: Give me a pop-quiz question on these slides!"
      : text.trim();

    if (!displayUserMsg) return;
    setChatInput("");

    setChatHistories((prev) => ({
      ...prev,
      [hash]: [
        ...(prev[hash] || []),
        { id: `u-${Date.now()}`, sender: "user", text: displayUserMsg },
      ],
    }));

    setIsAiTyping(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/chat`, {
        file_hash: hash,
        message: displayUserMsg,
        is_challenge_me: isChallengeMe,
      });

      setChatHistories((prev) => ({
        ...prev,
        [hash]: [
          ...(prev[hash] || []),
          {
            id: `ai-${Date.now()}`,
            sender: "ai",
            text: res.data.reply,
            citations: res.data.citations,
          },
        ],
      }));

      if (res.data.knowledge_summary) {
        setKnowledgeSummaries((prev) => ({
          ...prev,
          [hash]: res.data.knowledge_summary,
        }));
      }
    } catch {
      setChatHistories((prev) => ({
        ...prev,
        [hash]: [
          ...(prev[hash] || []),
          {
            id: `err-${Date.now()}`,
            sender: "ai",
            text: "Could not reach Socratic RAG endpoint.",
          },
        ],
      }));
    } finally {
      setIsAiTyping(false);
    }
  };

  const mediaUrl = activeDoc?.generated_media_path
    ? activeDoc.generated_media_path.startsWith("http")
      ? activeDoc.generated_media_path
      : `${API_BASE_URL}${activeDoc.generated_media_path}`
    : null;

  const ytValidation = activeDoc?.routing_payload?.youtube_validation;
  const videoPlaylist: ValidatedVideoItem[] =
    ytValidation?.playlist && ytValidation.playlist.length > 0
      ? ytValidation.playlist
      : ytValidation?.embed_url
      ? [
          {
            topic_title:
              activeDoc?.routing_payload?.youtube_search_queries?.[0] ||
              "Module 1: Primary Lecture Walkthrough",
            query:
              activeDoc?.routing_payload?.youtube_search_queries?.[0] ||
              "Lecture Topic",
            video_id: ytValidation.video_id,
            embed_url: ytValidation.embed_url,
            validation_summary: ytValidation.validation_summary,
          },
        ]
      : [];

  const currentTabIndex = activeDoc
    ? activeVideoTab[activeDoc.file_hash] || 0
    : 0;
  const currentVideo =
    videoPlaylist[currentTabIndex] || videoPlaylist[0] || null;

  return (
    <div className="min-h-screen w-full flex bg-[#eef4f9]">
      {/* =================================================================
          1. LEFT SIDEBAR: Brand Logo, Upload Dropzone & Queue State Machine
         ================================================================= */}
      <aside className="w-72 shrink-0 bg-gradient-to-b from-[#2c4f6b] via-[#1c3751] to-[#112437] text-white flex flex-col p-4 select-none border-r border-slate-700/40">
        <div className="mb-5 pt-1 px-1 flex items-baseline gap-2">
          <span className="text-xl font-extrabold tracking-wider text-white">
            NORTH STAR
          </span>
          <span
            className="text-2xl text-sky-300 relative inline-block"
            style={{ fontFamily: "var(--font-dancing), cursive" }}
          >
            learning
            <svg
              viewBox="0 0 24 24"
              fill="currentColor"
              className="w-3.5 h-3.5 text-sky-300 absolute -top-1.5 -right-3"
            >
              <path d="M12 0L14.5 9.5L24 12L14.5 14.5L12 24L9.5 14.5L0 12L9.5 9.5L12 0Z" />
            </svg>
          </span>
        </div>

        {/* Collapsible Upload Zone */}
        <div className="mb-5">
          <div className="flex items-center justify-between mb-2 px-1">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-sky-200/80">
              Ingest Lecture PDF
            </span>
            <button
              onClick={() => setIsUploadCollapsed(!isUploadCollapsed)}
              className="text-sky-200/80 hover:text-white p-1"
            >
              {isUploadCollapsed ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronUp className="w-4 h-4" />
              )}
            </button>
          </div>

          {!isUploadCollapsed ? (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                const f = e.dataTransfer.files?.[0];
                if (f?.name.toLowerCase().endsWith(".pdf")) setSelectedFile(f);
              }}
              className={`border-2 border-dashed rounded-xl p-4 text-center transition ${
                isDragging
                  ? "border-sky-300 bg-sky-400/20"
                  : "border-sky-300/40 bg-white/5 hover:bg-white/10"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) setSelectedFile(f);
                }}
                className="hidden"
              />
              <div
                onClick={() => fileInputRef.current?.click()}
                className="cursor-pointer flex flex-col items-center"
              >
                <CloudUpload className="w-8 h-8 text-sky-200 mb-2" />
                <p className="text-xs text-sky-100 mb-3 break-all">
                  {selectedFile
                    ? selectedFile.name
                    : "Drag & drop PDF here or click to browse"}
                </p>
              </div>
              <button
                onClick={handleUpload}
                disabled={isUploading}
                className="w-full py-2 px-3 bg-white text-[#1a334a] font-semibold text-xs rounded-lg shadow hover:bg-sky-50 transition disabled:opacity-60"
              >
                {isUploading
                  ? "Uploading..."
                  : selectedFile
                  ? "Start Pipeline"
                  : "Upload PDF"}
              </button>
            </div>
          ) : (
            <button
              onClick={() => setIsUploadCollapsed(false)}
              className="w-full py-2 px-3 rounded-xl border border-dashed border-sky-300/40 bg-white/5 hover:bg-white/10 text-xs font-medium text-sky-100 flex items-center justify-center gap-2"
            >
              <CloudUpload className="w-4 h-4 text-sky-300" />
              <span>+ Upload Another PDF</span>
            </button>
          )}
        </div>

        {/* My Documents Queue */}
        <div className="flex-1 flex flex-col min-h-0">
          <h2 className="text-xs font-bold uppercase tracking-wider text-sky-200/80 mb-2.5 px-1">
            My Documents ({documents.length})
          </h2>
          <div className="space-y-2.5 overflow-y-auto pr-1 flex-1">
            {documents.map((doc) => {
              const isSelected = doc.file_hash === activeHash;
              const isRunning =
                doc.status !== "COMPLETE" && doc.status !== "ERROR";

              return (
                <div
                  key={doc.file_hash}
                  onClick={() => setActiveHash(doc.file_hash)}
                  className={`rounded-xl p-3 transition cursor-pointer border ${
                    isSelected
                      ? "bg-white/15 border-sky-300/60 shadow-md"
                      : "bg-white/5 border-white/10 hover:bg-white/10"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <FileText className="w-4 h-4 text-sky-300 shrink-0" />
                      <span className="text-xs font-semibold text-white truncate">
                        {doc.filename}
                      </span>
                    </div>
                    <span className="shrink-0 text-[10px] font-bold uppercase px-2 py-0.5 rounded-full bg-slate-900/50 text-sky-200 border border-sky-300/20">
                      {doc.status === "COMPLETE"
                        ? doc.taxonomy_classification
                        : doc.status}
                    </span>
                  </div>

                  {isRunning && (
                    <div className="mt-2.5">
                      <div className="w-full h-1.5 bg-slate-900/60 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-sky-400 to-cyan-300 transition-all duration-500"
                          style={{ width: `${doc.progress}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-[10px] text-sky-200/80 mt-1">
                        <span>Polling every 5s...</span>
                        <span>{doc.progress}%</span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </aside>

      {/* =================================================================
          2. DUAL-PANE MAIN WORKSPACE (Left: Original PDF | Right: Media + RAG)
         ================================================================= */}
      {!activeDoc ? (
        <main className="flex-1 flex flex-col items-center justify-center p-8 text-center">
          <div className="w-16 h-16 rounded-2xl bg-sky-100 text-[#234766] flex items-center justify-center mb-4">
            <BookOpen className="w-8 h-8" />
          </div>
          <h1 className="text-xl font-bold text-slate-800 mb-2">
            Dual-Pane Active Learning Workspace
          </h1>
          <p className="text-sm text-slate-500 max-w-md">
            Upload a lecture PDF on the left. Your original slide deck will mount
            on the left pane while the AI-generated video, podcast, or validated
            proof sandbox mounts on the right.
          </p>
        </main>
      ) : (
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 h-screen overflow-hidden">
          {/* -------------------------------------------------------------
              LEFT PANE (5 Cols): Original PDF Viewer & Raw LaTeX Toggle
             ------------------------------------------------------------- */}
          <section className="lg:col-span-5 bg-white border-r border-slate-200 flex flex-col h-screen overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-200 flex items-center justify-between bg-slate-50">
              <div className="flex items-center gap-2 min-w-0">
                <FileText className="w-4 h-4 text-[#234766] shrink-0" />
                <span className="text-xs font-bold text-slate-800 truncate">
                  {activeDoc.filename}
                </span>
              </div>
              <div className="flex rounded-lg bg-slate-200/80 p-0.5 text-[11px] font-semibold">
                <button
                  onClick={() => setLeftPaneMode("pdf")}
                  className={`px-2.5 py-1 rounded-md transition ${
                    leftPaneMode === "pdf"
                      ? "bg-white text-slate-900 shadow-2xs"
                      : "text-slate-600"
                  }`}
                >
                  Original PDF
                </button>
                <button
                  onClick={() => setLeftPaneMode("markdown")}
                  className={`px-2.5 py-1 rounded-md transition ${
                    leftPaneMode === "markdown"
                      ? "bg-white text-slate-900 shadow-2xs"
                      : "text-slate-600"
                  }`}
                >
                  Extracted LaTeX
                </button>
              </div>
            </div>

            <div className="flex-1 bg-slate-800 overflow-hidden">
              {leftPaneMode === "pdf" && activeDoc.pdf_url ? (
                <iframe
                  src={`${API_BASE_URL}${activeDoc.pdf_url}#toolbar=0&navpanes=0`}
                  className="w-full h-full border-0"
                  title="Original Lecture PDF"
                />
              ) : (
                <div className="p-5 h-full overflow-y-auto bg-slate-900 text-slate-100 font-mono text-xs whitespace-pre-wrap leading-relaxed">
                  {activeDoc.raw_extracted_text ||
                    "Extracting chronological Markdown and LaTeX from slides..."}
                </div>
              )}
            </div>
          </section>

          {/* -------------------------------------------------------------
              RIGHT PANE (7 Cols): Dynamic Media Player + Sandbox + Socratic RAG
             ------------------------------------------------------------- */}
          <section className="lg:col-span-7 bg-[#f4f8fb] h-screen overflow-y-auto p-6 space-y-6">
            {/* Top Status Header */}
            <div className="shrink-0 flex flex-wrap items-center justify-between gap-3 bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
              <div>
                <div className="flex items-center gap-2.5">
                  <h1 className="text-lg font-extrabold text-slate-900">
                    {activeDoc.filename.replace(/\.pdf$/i, "")}
                  </h1>
                  {activeDoc.taxonomy_classification && (
                    <span className="px-3 py-0.5 rounded-full text-[11px] font-bold uppercase tracking-wider bg-[#234766] text-white">
                      {activeDoc.taxonomy_classification} PATH
                    </span>
                  )}
                </div>
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {activeDoc.routing_payload?.core_concepts?.map((c, i) => (
                    <span
                      key={i}
                      className="px-2.5 py-0.5 rounded-full bg-[#e5eef7] text-[#1e3a52] text-[11px] font-semibold"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Wait State if Phase 3 Media is Still Generating */}
            {activeDoc.status !== "COMPLETE" && activeDoc.status !== "ERROR" && (
              <div className="shrink-0 bg-white rounded-2xl p-8 border border-slate-200/80 text-center space-y-3">
                <div className="w-10 h-10 rounded-full border-4 border-sky-200 border-t-[#234766] animate-spin mx-auto" />
                <h2 className="text-sm font-bold text-slate-800">
                  Phase 3 Worker Active: Status is {activeDoc.status}
                </h2>
                <p className="text-xs text-slate-500 max-w-md mx-auto">
                  Background worker is validating transcripts for all lecture
                  modules, generating media, and indexing slide embeddings...
                </p>
              </div>
            )}

            {/* ===========================================================
                DYNAMIC MEDIA MOUNTING: PATH A (LOGIC) -> Tabbed Playlist + Sandbox
               =========================================================== */}
            {activeDoc.taxonomy_classification === "LOGIC" &&
              activeDoc.routing_payload && (
                <div className="shrink-0 space-y-5">
                  {/* Multi-Topic Tabbed YouTube Player */}
                  <div className="bg-white rounded-2xl p-4 border border-slate-200/80 shadow-xs space-y-3.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
                        <PlayCircle className="w-5 h-5 text-rose-600" />
                        <span>
                          Syllabus-Validated Video Modules ({videoPlaylist.length})
                        </span>
                      </div>
                      <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                        Transcript Verified
                      </span>
                    </div>

                    {videoPlaylist.length > 1 && (
                      <div className="flex items-center gap-2 overflow-x-auto pb-1">
                        {videoPlaylist.map((vid, idx) => {
                          const isTabActive = idx === currentTabIndex;
                          return (
                            <button
                              key={vid.video_id + idx}
                              onClick={() =>
                                setActiveVideoTab((prev) => ({
                                  ...prev,
                                  [activeDoc.file_hash]: idx,
                                }))
                              }
                              className={`px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition border ${
                                isTabActive
                                  ? "bg-[#1b3650] text-white border-[#1b3650] shadow-xs"
                                  : "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100"
                              }`}
                            >
                              Module {idx + 1}: {vid.query}
                            </button>
                          );
                        })}
                      </div>
                    )}

                    {currentVideo ? (
                      <>
                        <div className="aspect-video w-full rounded-xl overflow-hidden bg-slate-950">
                          <iframe
                            key={currentVideo.embed_url}
                            src={currentVideo.embed_url}
                            className="w-full h-full border-0"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowFullScreen
                            title={currentVideo.topic_title}
                          />
                        </div>

                        <div className="text-xs text-slate-600 bg-slate-50 p-3 rounded-xl border border-slate-200/60 flex items-start justify-between gap-3">
                          <div>
                            <strong className="text-slate-800">
                              NLP Transcript Audit ({currentVideo.query}):{" "}
                            </strong>
                            <Latex>{currentVideo.validation_summary}</Latex>
                          </div>
                          <a
                            href={`https://www.youtube.com/watch?v=${currentVideo.video_id}`}
                            target="_blank"
                            rel="noreferrer"
                            className="shrink-0 text-[11px] font-bold text-[#234766] hover:underline"
                          >
                            Open on YouTube &rarr;
                          </a>
                        </div>
                      </>
                    ) : null}
                  </div>

                  {/* Active Practice Sandbox */}
                  {activeDoc.routing_payload.practice_sandbox && (
                    <div className="bg-white rounded-2xl p-5 border-2 border-[#3b6990] shadow-sm space-y-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {sandboxUnlocked[activeDoc.file_hash] ? (
                            <Unlock className="w-5 h-5 text-emerald-600" />
                          ) : (
                            <Lock className="w-5 h-5 text-amber-600" />
                          )}
                          <h2 className="text-sm font-extrabold text-slate-900">
                            Active Practice Sandbox (Required to Unlock Mastery)
                          </h2>
                        </div>
                        <span
                          className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full ${
                            sandboxUnlocked[activeDoc.file_hash]
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-amber-100 text-amber-800"
                          }`}
                        >
                          {sandboxUnlocked[activeDoc.file_hash]
                            ? "UNLOCKED / PASSED"
                            : "PROGRESSION LOCKED"}
                        </span>
                      </div>

                      <div className="text-xs text-slate-700 leading-relaxed bg-slate-50 p-3.5 rounded-xl border border-slate-200 prose prose-sm max-w-none">
                        <Latex>
                          {
                            activeDoc.routing_payload.practice_sandbox
                              .question_markdown
                          }
                        </Latex>
                      </div>

                      <textarea
                        rows={4}
                        value={sandboxInput[activeDoc.file_hash] || ""}
                        onChange={(e) =>
                          setSandboxInput((prev) => ({
                            ...prev,
                            [activeDoc.file_hash]: e.target.value,
                          }))
                        }
                        placeholder="Write your formal proof step or code solution here..."
                        className="w-full font-mono text-xs bg-slate-900 text-sky-100 rounded-xl p-3.5 border border-slate-700 focus:outline-none focus:border-sky-400"
                      />

                      <div className="flex items-center justify-between gap-3">
                        <button
                          type="button"
                          onClick={() => setShowHint(!showHint)}
                          className="text-xs font-semibold text-slate-500 hover:text-slate-800 flex items-center gap-1"
                        >
                          <HelpCircle className="w-4 h-4" />
                          <span>{showHint ? "Hide Hint" : "Need a Hint?"}</span>
                        </button>

                        <button
                          type="button"
                          onClick={handleVerifySandbox}
                          disabled={isVerifyingSandbox}
                          className="px-4 py-2 bg-[#1b3650] hover:bg-[#284b6b] text-white text-xs font-bold rounded-xl transition disabled:opacity-50"
                        >
                          {isVerifyingSandbox
                            ? "Grading Proof/Code..."
                            : "Submit & Verify Challenge"}
                        </button>
                      </div>

                      {showHint && (
                        <div className="text-xs text-amber-900 bg-amber-50 border border-amber-200 p-3 rounded-xl">
                          <strong>Hint: </strong>
                          <Latex>
                            {activeDoc.routing_payload.practice_sandbox.hint}
                          </Latex>
                        </div>
                      )}

                      {sandboxFeedback[activeDoc.file_hash] && (
                        <div
                          className={`text-xs p-3.5 rounded-xl border ${
                            sandboxUnlocked[activeDoc.file_hash]
                              ? "bg-emerald-50 border-emerald-200 text-emerald-900"
                              : "bg-rose-50 border-rose-200 text-rose-900"
                          }`}
                        >
                          <strong>AI Evaluator Feedback: </strong>
                          <Latex>{sandboxFeedback[activeDoc.file_hash]}</Latex>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

            {/* ===========================================================
                DYNAMIC MEDIA MOUNTING: PATH B (SPATIAL) -> Video/Audio + Scenes
               =========================================================== */}
            {activeDoc.taxonomy_classification === "SPATIAL" &&
              activeDoc.routing_payload && (
                <div className="shrink-0 bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
                      <Video className="w-5 h-5 text-[#234766]" />
                      <span>Spatial Visual &amp; Audio Explainer</span>
                    </div>
                    <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-sky-50 text-[#234766] border border-sky-200">
                      Synchronized Slide Visualizer
                    </span>
                  </div>

                  {mediaUrl && mediaUrl.endsWith(".mp4") ? (
                    <video
                      src={mediaUrl}
                      controls
                      className="w-full rounded-xl bg-slate-950 aspect-video shadow-md"
                    />
                  ) : mediaUrl &&
                    (mediaUrl.endsWith(".mp3") || mediaUrl.endsWith(".wav")) ? (
                    <div className="bg-gradient-to-r from-[#1c3751] to-[#2c5373] text-white p-4 rounded-xl space-y-2">
                      <div className="flex items-center gap-2 text-xs font-semibold text-sky-200">
                        <Volume2 className="w-4 h-4" />
                        <span>Synchronized Spatial Audio Walkthrough</span>
                      </div>
                      <audio src={mediaUrl} controls className="w-full h-9" />
                    </div>
                  ) : null}

                  <div className="text-xs text-slate-700 whitespace-pre-line leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-200/60 max-h-64 overflow-y-auto prose prose-sm max-w-none">
                    <Latex>
                      {activeDoc.routing_payload.spatial_explainer_prompt}
                    </Latex>
                  </div>
                </div>
              )}

            {/* ===========================================================
                DYNAMIC MEDIA MOUNTING: PATH C (ABSTRACT) -> Audio Podcast Player
               =========================================================== */}
            {activeDoc.taxonomy_classification === "ABSTRACT" &&
              activeDoc.routing_payload && (
                <div className="shrink-0 bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs space-y-4">
                  <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
                    <Volume2 className="w-5 h-5 text-[#234766]" />
                    <span>Generated Theoretical Synthesis Podcast</span>
                  </div>

                  {mediaUrl && (
                    <div className="bg-gradient-to-r from-[#1c3751] to-[#2c5373] p-4 rounded-xl">
                      <audio src={mediaUrl} controls className="w-full" />
                    </div>
                  )}

                  <div className="text-xs text-slate-700 whitespace-pre-line leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-200/60 max-h-60 overflow-y-auto">
                    {activeDoc.routing_payload.abstract_podcast_script}
                  </div>
                </div>
              )}

            {/* ===========================================================
                PHASE 5: SOCRATIC RAG CHAT + ASYMMETRIC MEMORY BUFFER
               =========================================================== */}
            <div className="shrink-0 bg-white rounded-2xl border border-slate-200/80 shadow-xs flex flex-col overflow-hidden mb-8">
              <div className="px-5 py-3.5 border-b border-slate-100 flex flex-wrap items-center justify-between gap-2 bg-slate-50/60">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-full bg-[#e5eef7] text-[#1e3a52] flex items-center justify-center">
                    <BrainCircuit className="w-4 h-4" />
                  </div>
                  <div>
                    <h2 className="text-sm font-bold text-slate-900">
                      Socratic RAG Tutor (Asymmetric Memory)
                    </h2>
                    <p className="text-[11px] text-slate-500">
                      Buffers last 4 turns + vector slide retrieval • Never gives direct answers
                    </p>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => sendSocraticMessage("", true)}
                  disabled={isAiTyping}
                  className="px-3 py-1.5 rounded-xl bg-amber-500 hover:bg-amber-600 text-white text-xs font-bold shadow-2xs transition flex items-center gap-1.5 disabled:opacity-50"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Challenge Me (Pop Quiz)</span>
                </button>
              </div>

              {knowledgeSummaries[activeDoc.file_hash] && (
                <div className="px-4 py-2 bg-sky-50/90 border-b border-sky-100 text-[11px] text-[#1e3a52] flex items-center gap-2">
                  <span className="font-bold uppercase tracking-wider text-[10px] px-2 py-0.5 rounded bg-sky-200/70">
                    Knowledge State
                  </span>
                  <span className="truncate">
                    {knowledgeSummaries[activeDoc.file_hash]}
                  </span>
                </div>
              )}

              <div className="p-4 space-y-3 max-h-80 overflow-y-auto bg-slate-50/30">
                {(chatHistories[activeDoc.file_hash] || []).map((msg) =>
                  msg.sender === "ai" ? (
                    <div key={msg.id} className="flex items-start gap-2.5">
                      <div className="w-7 h-7 rounded-full bg-[#234766] text-white flex items-center justify-center shrink-0 mt-1 text-[11px] font-bold">
                        AI
                      </div>
                      <div className="bg-[#eef3f8] text-slate-800 rounded-2xl rounded-tl-xs p-3.5 text-xs leading-relaxed border border-slate-200/60 max-w-[88%]">
                        <div className="prose prose-sm max-w-none">
                          <Latex>{msg.text}</Latex>
                        </div>
                        {msg.citations && msg.citations.length > 0 && (
                          <div className="flex flex-wrap gap-1.5 mt-2">
                            {msg.citations.map((cit, i) => (
                              <span
                                key={i}
                                className="text-[10px] font-semibold px-2 py-0.5 rounded bg-white text-[#234766] border border-slate-200"
                              >
                                📌 Retrieved: {cit}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div key={msg.id} className="flex justify-end">
                      <div className="bg-[#1b3650] text-white rounded-2xl rounded-tr-xs p-3 text-xs leading-relaxed max-w-[80%]">
                        <Latex>{msg.text}</Latex>
                      </div>
                    </div>
                  )
                )}

                {isAiTyping && (
                  <div className="text-xs text-slate-400 pl-9 italic">
                    Retrieving slide vectors &amp; formulating Socratic question...
                  </div>
                )}
              </div>

              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  sendSocraticMessage(chatInput, false);
                }}
                className="p-3 border-t border-slate-200/80 bg-white flex items-center gap-2"
              >
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Respond to the Socratic tutor or ask about a slide..."
                  className="flex-1 text-xs bg-slate-50 border border-slate-200 rounded-xl px-3.5 py-2.5 text-slate-800 focus:outline-none focus:border-[#3b6990]"
                />
                <button
                  type="submit"
                  disabled={!chatInput.trim() || isAiTyping}
                  className="w-9 h-9 rounded-xl bg-[#1b3650] hover:bg-[#284b6b] text-white flex items-center justify-center transition disabled:opacity-40"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}