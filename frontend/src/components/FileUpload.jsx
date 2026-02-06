import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Image, Loader2 } from 'lucide-react'

function FileUpload({ onUpload, loading }) {
    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles.length > 0) {
            onUpload(acceptedFiles[0])
        }
    }, [onUpload])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.png', '.jpg', '.jpeg', '.bmp']
        },
        maxFiles: 1,
        disabled: loading
    })

    return (
        <div className="gradient-border p-[2px] rounded-2xl">
            <div
                {...getRootProps()}
                className={`
          glass-dark rounded-2xl p-12 cursor-pointer
          transition-all duration-300 ease-out
          ${isDragActive ? 'upload-zone active scale-[1.02]' : 'upload-zone'}
          ${loading ? 'opacity-60 cursor-not-allowed' : 'hover:scale-[1.01]'}
        `}
            >
                <input {...getInputProps()} />

                <div className="text-center">
                    {loading ? (
                        <>
                            <div className="relative w-20 h-20 mx-auto mb-6">
                                <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 animate-spin" />
                                <div className="absolute inset-1 rounded-full bg-slate-900 flex items-center justify-center">
                                    <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
                                </div>
                            </div>
                            <h3 className="text-xl font-semibold text-white mb-2">
                                Analyzing Fingerprint...
                            </h3>
                            <p className="text-slate-400">
                                Running deep learning model
                            </p>
                        </>
                    ) : (
                        <>
                            <div className={`
                w-20 h-20 mx-auto mb-6 rounded-2xl
                bg-gradient-to-br from-blue-500/20 to-purple-500/20
                border border-white/10 flex items-center justify-center
                transition-transform duration-300
                ${isDragActive ? 'scale-110' : ''}
              `}>
                                {isDragActive ? (
                                    <Image className="w-10 h-10 text-blue-400 animate-pulse" />
                                ) : (
                                    <Upload className="w-10 h-10 text-blue-400" />
                                )}
                            </div>

                            <h3 className="text-xl font-semibold text-white mb-2">
                                {isDragActive ? 'Drop your fingerprint image' : 'Upload Fingerprint Image'}
                            </h3>

                            <p className="text-slate-400 mb-4">
                                Drag and drop or click to select
                            </p>

                            <div className="flex items-center justify-center gap-2 text-sm text-slate-500">
                                <span className="px-2 py-1 rounded bg-white/5">PNG</span>
                                <span className="px-2 py-1 rounded bg-white/5">JPG</span>
                                <span className="px-2 py-1 rounded bg-white/5">BMP</span>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}

export default FileUpload
