#include "agent_bridge.h"
#include <windows.h>
#include <winhttp.h>
#include <shellapi.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <thread>
using json=nlohmann::json;
namespace fs=std::filesystem;
namespace {
std::string utf8(const fs::path& p){auto s=p.u8string();return {s.begin(),s.end()};}
struct Session { unsigned short port; std::string token,url; fs::path data; };
std::string percent(const std::string& s){std::string out; const char* hex="0123456789ABCDEF";for(unsigned char c:s){if(isalnum(c)||c=='.'||c=='-'||c=='_')out+=char(c);else{out+='%';out+=hex[c>>4];out+=hex[c&15];}}return out;}
Session session(){
 wchar_t dir[32768]{};DWORD n=GetEnvironmentVariableW(L"IBP_AGENT_DATA",dir,32768);
 fs::path folder;
 if(n&&n<32768)folder=dir;else{if(!GetEnvironmentVariableW(L"LOCALAPPDATA",dir,32768))throw std::runtime_error("LOCALAPPDATA unavailable");folder=fs::path(dir)/"IBPPhotoAgent";}
 std::ifstream in(folder/"session.json");if(!in)throw std::runtime_error("请从 IBP Photo Agent 启动 Studio");
 auto s=json::parse(in);auto url=s.at("url").get<std::string>();auto mark=url.find('#');
 auto port=s.at("port").get<int>();if(mark==std::string::npos||port<1||port>65535)throw std::runtime_error("Agent session invalid");
 return {static_cast<unsigned short>(port),url.substr(mark+1),url,folder};
}
struct Http {HINTERNET h{};explicit Http(HINTERNET v):h(v){if(!h)throw std::runtime_error("Agent connection failed");}~Http(){WinHttpCloseHandle(h);}operator HINTERNET()const{return h;}};
json request(const Session& s,const wchar_t* method,const std::wstring& path,const std::string& body="",const std::wstring& extra=L""){
 Http client(WinHttpOpen(L"IBP-Studio/0.2",WINHTTP_ACCESS_TYPE_NO_PROXY,nullptr,nullptr,0));
 WinHttpSetTimeouts(client,5000,5000,15000,30000);
 Http conn(WinHttpConnect(client,L"127.0.0.1",s.port,0));
 Http req(WinHttpOpenRequest(conn,method,path.c_str(),nullptr,WINHTTP_NO_REFERER,WINHTTP_DEFAULT_ACCEPT_TYPES,0));
 DWORD redirects=WINHTTP_DISABLE_REDIRECTS;WinHttpSetOption(req,WINHTTP_OPTION_DISABLE_FEATURE,&redirects,sizeof(redirects));
 std::wstring headers=L"Content-Type: application/json\r\nX-IBP-Token: "+std::wstring(s.token.begin(),s.token.end())+L"\r\n"+extra;
 if(!WinHttpSendRequest(req,headers.c_str(),DWORD(-1),body.empty()?WINHTTP_NO_REQUEST_DATA:(void*)body.data(),DWORD(body.size()),DWORD(body.size()),0)||!WinHttpReceiveResponse(req,nullptr))throw std::runtime_error("Agent 未连接");
 DWORD status=0,len=sizeof(status);WinHttpQueryHeaders(req,WINHTTP_QUERY_STATUS_CODE|WINHTTP_QUERY_FLAG_NUMBER,nullptr,&status,&len,nullptr);
 std::string data;char buf[16384];DWORD read=0;while(WinHttpReadData(req,buf,sizeof(buf),&read)&&read){data.append(buf,read);if(data.size()>4*1024*1024)throw std::runtime_error("Agent response too large");}
 auto result=json::parse(data);if(status>=400)throw std::runtime_error(result.value("error","Agent request failed"));return result;
}
}
AgentResult agentGenerate(const fs::path& source,const std::string& prompt,bool teacher){
 AgentResult out;out.source=source;
 try {
  auto s=session();if(!request(s,L"GET",L"/api/status").value("ready",false))throw std::runtime_error("模型加载中，请稍后重试");
  auto size=fs::file_size(source);if(size>100*1024*1024)throw std::runtime_error("单张文件上限 100 MB");
  std::ifstream in(source,std::ios::binary);if(!in)throw std::runtime_error("无法读取原图");std::string bytes((std::istreambuf_iterator<char>(in)),{});
  auto name=percent(utf8(source.filename()));auto upload=request(s,L"POST",L"/api/upload",bytes,L"X-Filename: "+std::wstring(name.begin(),name.end())+L"\r\n");
  auto queued=request(s,L"POST",L"/api/jobs",json{{"upload",upload.at("id")},{"prompt",prompt},{"role",teacher?"teacher":"student"}}.dump());
  out.job=queued.at("id").get<std::string>();auto route=L"/api/jobs/"+std::wstring(out.job.begin(),out.job.end());
  for(int i=0;i<900;i++){
   auto job=request(s,L"GET",route);auto status=job.value("status","");
   if(status=="error")throw std::runtime_error(job.value("error","生成失败"));
   if(status=="done"){
    auto folder=s.data/"jobs"/out.job;out.recipe=folder/"preview.imgarg";out.preview=folder/"after.jpg";out.before=folder/"before.jpg";return out;
   }
   std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  throw std::runtime_error("生成超时；可在 Agent 工作台继续查看");
 }catch(const std::exception& e){out.error=e.what();return out;}
}
void agentFeedback(const std::string& job,bool accepted,const std::string& group){request(session(),L"POST",L"/api/feedback",json{{"id",job},{"accepted",accepted},{"group_id",group}}.dump());}
void agentOpenPanel(const char* panel){auto s=session();auto hash=s.url.find('#');auto url=s.url.substr(0,hash)+"?panel="+panel+"#"+s.token;std::wstring wide(url.begin(),url.end());if(reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr,L"open",wide.c_str(),nullptr,nullptr,SW_SHOWNORMAL))<=32)throw std::runtime_error("无法打开 Agent 工作台");}
