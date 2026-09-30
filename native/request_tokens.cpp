// Use the serving frontend's exact prompt renderer and native tokenizer.
#define main q27_frontend_main
#include "q27_fe_rewind.cpp"
#undef main
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    hf::Tokenizer tokenizer(std::filesystem::path(argv[1])/"tokenizer.json");
    std::string line;
    while(std::getline(std::cin,line)) {
        bool ok=false;const auto req=jz::parse(line,&ok);if(!ok)return 3;
        std::vector<ToolDef> defs;
        const auto& tools=req["tools"];
        if(tools.is_arr())for(size_t i=0;i<tools.size();++i){ToolDef t;t.schema=tools.at(i);t.name=t.schema["function"]["name"].as_str();t.desc=t.schema["function"]["description"].as_str();defs.push_back(t);}
        const auto prompt=render_prompt(req["messages"],defs,req["chat_template_kwargs"]["enable_thinking"].as_bool(true),req["reasoning_effort"].as_str());
        std::cout<<tokenizer.encode(prompt).size()<<std::endl;
    }
}
