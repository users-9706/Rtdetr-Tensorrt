#include "logger.h"
#include "ErrorRecorder.h"
#include "logging.h"
using namespace nvinfer1;
SampleErrorRecorder gRecorder;
namespace sample
{
    Logger gLogger{Logger::Severity::kINFO};
    LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
    LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
    LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
    LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
    LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};
    void setReportableSeverity(Logger::Severity severity)
    {
        gLogger.setReportableSeverity(severity);
        gLogVerbose.setReportableSeverity(severity);
        gLogInfo.setReportableSeverity(severity);
        gLogWarning.setReportableSeverity(severity);
        gLogError.setReportableSeverity(severity);
        gLogFatal.setReportableSeverity(severity);
    }
} 
