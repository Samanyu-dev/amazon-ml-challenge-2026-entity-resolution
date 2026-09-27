// Local, bounded-memory multi-view inverted-index candidate generator.
// Newly authored implementation: MIT License (see packaged LICENSE).
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <filesystem>
using namespace std;
static constexpr int NF=54;
static const vector<string> FN={
 "name_edit","name_trigram_dice","name_sorted_edit","name_token_jaccard","name_token_min","name_token_max","name_exact","name_compact_exact","full_name_edit","name_length_ratio","name_idf_overlap","phonetic_edit","phonetic_token_overlap",
 "address_edit","address_trigram_dice","address_token_jaccard","address_token_min","address_token_max","address_idf_overlap","number_jaccard","number_min","both_have_numbers","first_number_equal","first_number_conflict","target_address_missing","anchor_name_nonascii","target_name_nonascii","anchor_name_words","target_name_words","anchor_address_words","target_address_words","first_name_word_edit","name_token_fuzzy_mean","name_token_fuzzy_min","name_token_fuzzy_max","address_length_ratio","address_exact","name_address_product","name_address_min","retrieval_name_score","retrieval_address_score","retrieval_combined","retrieval_rank","retrieval_best_margin","same_first_address_token","same_last_address_token","name_shared_words","address_shared_words","source_is_s3","name_core_empty","phonetic_compact_exact","number_count_difference","name_frequency_log","raw_name_token_jaccard"};
struct Rec {string eid,country,n,c,a,p;int script=0,missing=0,owner=-1,fold=-1;};
vector<string> split(const string&s,char delim=' '){vector<string> v;size_t st=0;for(size_t i=0;i<=s.size();i++)if(i==s.size()||s[i]==delim){if(i>st || delim=='\t')v.emplace_back(s.substr(st,i-st));st=i+1;}return v;}
Rec parse(const string&s){auto f=split(s,'\t');Rec r;if(f.size()<10)throw runtime_error("bad normalized row");r.eid=move(f[0]);r.country=move(f[1]);r.n=move(f[2]);r.c=move(f[3]);r.a=move(f[4]);r.p=move(f[5]);r.script=stoi(f[6]);r.missing=stoi(f[7]);r.owner=stoi(f[8]);r.fold=stoi(f[9]);return r;}
uint64_t hashstr(const string&s,uint64_t seed=1469598103934665603ULL){uint64_t h=seed;for(unsigned char c:s){h^=c;h*=1099511628211ULL;}return h;}
string compact(const string&s){string o;for(char c:s)if(c!=' ')o+=c;return o;}
vector<string> toks(const string&s){auto v=split(s);sort(v.begin(),v.end());v.erase(unique(v.begin(),v.end()),v.end());return v;}
vector<uint32_t> grams(const string&s,int n=3){vector<uint32_t> v;if(s.empty())return v;string z=" "+s+" ";for(size_t i=0;i+n<=z.size();i++){uint32_t h=0;for(int j=0;j<n;j++)h=h*131+(unsigned char)z[i+j];v.push_back(h);}sort(v.begin(),v.end());v.erase(unique(v.begin(),v.end()),v.end());return v;}
template<class T> int intersection(const vector<T>&a,const vector<T>&b){size_t i=0,j=0;int n=0;while(i<a.size()&&j<b.size()){if(a[i]==b[j]){n++;i++;j++;}else if(a[i]<b[j])i++;else j++;}return n;}
float div0(float a,float b){return b>0?a/b:0;}
float dice(const vector<uint32_t>&a,const vector<uint32_t>&b){return div0(2.f*intersection(a,b),a.size()+b.size());}
float leng(const string&a,const string&b){return div0(min(a.size(),b.size()),max(a.size(),b.size()));}
template<class U> float myers(const string&a,const string&b){int m=a.size();if(!m)return b.empty()?1:0;array<U,128> peq{};for(int i=0;i<m;i++)peq[(unsigned char)a[i]&127]|=U(1)<<i;U pv=~U(0),mv=0,high=U(1)<<(m-1);int score=m;for(unsigned char ch:b){U eq=peq[ch&127],xv=eq|mv,xh=(((eq&pv)+pv)^pv)|eq;U ph=mv|~(xh|pv),mh=pv&xh;if(ph&high)score++;if(mh&high)score--;ph=(ph<<1)|1;mh<<=1;pv=mh|~(xv|ph);mv=ph&xv;}return 1.f-float(score)/max(a.size(),b.size());}
float edit(const string&a,const string&b){if(a==b)return a.empty()?0:1;if(a.empty()||b.empty())return 0;const string&x=a.size()<b.size()?a:b;const string&y=a.size()<b.size()?b:a;if(x.size()<=63)return myers<uint64_t>(x,y);if(x.size()<=127)return myers<__uint128_t>(x,y);return dice(grams(a),grams(b));}
string joined(const vector<string>&v){string s;for(auto&w:v){if(!s.empty())s+=' ';s+=w;}return s;}
string firstnum(const string&s){string o;bool in=false;for(char c:s){if(c>='0'&&c<='9'){o+=c;in=true;}else if(in)break;}if(o.empty())return o;auto p=o.find_first_not_of('0');return p==string::npos?"0":o.substr(p);}
vector<string> numbers(const string&s){vector<string> v;string x;for(size_t i=0;i<=s.size();i++){char c=i<s.size()?s[i]:' ';if(c>='0'&&c<='9')x+=c;else if(!x.empty()){auto p=x.find_first_not_of('0');v.push_back(p==string::npos?"0":x.substr(p));x.clear();}}sort(v.begin(),v.end());v.erase(unique(v.begin(),v.end()),v.end());return v;}
struct Term {uint32_t df=0;vector<uint32_t> post;};
struct Key {uint64_t h;int type;};
struct Index {
 vector<Rec> anchors;unordered_map<uint64_t,Term> inv;size_t country_n=1;
 uint64_t key(const string&country,int type,const string&t)const{return hashstr(t,hashstr(country)^((uint64_t(type)+1)*0x9e3779b97f4a7c15ULL));}
 vector<Key> keys(const Rec&r)const{
  vector<Key> k;auto add=[&](int type,const string&t){if(!t.empty())k.push_back({key(r.country,type,t),type});};
  add(0,compact(r.c));for(auto&w:toks(r.c))if(w.size()>=2)add(1,w);
  for(auto g:grams(r.c,3))add(2,to_string(g));
  for(auto&w:toks(r.a))if(w.size()>=2)add(3,w);
  for(auto g:grams(r.a,4))add(4,to_string(g));
  for(auto&w:toks(r.p))if(w.size()>=2)add(5,w);
  add(6,r.a);return k;
 }
 float idf(const string&country,int type,const string&t)const{auto it=inv.find(key(country,type,t));return it==inv.end()?0:log1p(float(anchors.size())/(1+it->second.df));}
 void build(const string&file){ifstream f(file);string line;uint32_t n=0;auto t=chrono::steady_clock::now();inv.reserve(3000000);anchors.reserve(2300000);while(getline(f,line)){Rec r=parse(line);anchors.push_back(move(r));for(auto k:keys(anchors.back())){auto&term=inv[k.h];term.df++;if(term.df<=40000)term.post.push_back(n);else if(term.df==40001)vector<uint32_t>().swap(term.post);}n++;if(n%200000==0)cerr<<"INDEX "<<n<<" terms="<<inv.size()<<"\n";}cerr<<"INDEX_COMPLETE anchors="<<n<<" terms="<<inv.size()<<" seconds="<<chrono::duration<double>(chrono::steady_clock::now()-t).count()<<"\n";}
};
struct Hit {uint32_t aid;float ns,as,rank;float nq=0,aq=0;};
struct Searcher{
 const Index&idx;vector<uint32_t> stamps,touched;vector<float> ns,as;uint32_t stamp=0;
 Searcher(const Index&i):idx(i),stamps(i.anchors.size(),0),ns(i.anchors.size(),0),as(i.anchors.size(),0){}
 vector<Hit> retrieve(const Rec&q,int preliminary){
  if(++stamp==0){fill(stamps.begin(),stamps.end(),0);stamp=1;}touched.clear();
  array<vector<pair<const Term*,int>>,7> lists;
  for(auto k:idx.keys(q)){auto it=idx.inv.find(k.h);if(it!=idx.inv.end()&&!it->second.post.empty())lists[k.type].push_back({&it->second,k.type});}
  const int maxterms[7]={1,6,12,9,10,6,1};const int budgets[7]={12000,16000,14000,18000,12000,10000,12000};
  const float wt[7]={6,2,.38f,1,.24f,1.0f,7};
  for(int ty=0;ty<7;ty++){
   auto&ls=lists[ty];sort(ls.begin(),ls.end(),[](auto a,auto b){return a.first->df<b.first->df;});int work=0,terms=0;
   for(auto [p,type]:ls){if(terms>=maxterms[ty])break;if(work+(int)p->post.size()>budgets[ty])continue;work+=p->post.size();terms++;float w=wt[ty]*log1p(float(idx.anchors.size())/(1+p->df));
    for(uint32_t aid:p->post){if(stamps[aid]!=stamp){stamps[aid]=stamp;ns[aid]=as[aid]=0;touched.push_back(aid);}if(ty==3||ty==4||ty==6)as[aid]+=w;else ns[aid]+=w;}
   }
  }
  vector<Hit> all;all.reserve(touched.size());float nmax=0,amax=0;for(auto aid:touched){nmax=max(nmax,ns[aid]);amax=max(amax,as[aid]);}
  for(auto aid:touched){float n=div0(ns[aid],nmax),a=div0(as[aid],amax);all.push_back({aid,ns[aid],as[aid],n+a+.65f*min(n,a)});}
  auto top=[&](auto cmp,int count){vector<Hit> v=all;count=min(count,(int)v.size());if(count<(int)v.size())nth_element(v.begin(),v.begin()+count,v.end(),cmp);v.resize(count);return v;};
  auto selected=top([](auto a,auto b){return a.rank>b.rank;},preliminary);
  for(auto h:top([](auto a,auto b){return a.ns>b.ns;},4))if(none_of(selected.begin(),selected.end(),[&](auto z){return z.aid==h.aid;}))selected.push_back(h);
  for(auto h:top([](auto a,auto b){return a.as>b.as;},4))if(none_of(selected.begin(),selected.end(),[&](auto z){return z.aid==h.aid;}))selected.push_back(h);
  return selected;
 }
};
float weighted_overlap(const Index&idx,const string&country,int ty,const vector<string>&a,const vector<string>&b){float sa=0,sb=0,si=0;for(auto&w:a){float z=idx.idf(country,ty,w);sa+=z;if(binary_search(b.begin(),b.end(),w))si+=z;}for(auto&w:b)sb+=idx.idf(country,ty,w);return div0(si,min(sa,sb));}
array<float,NF> features(const Index&idx,const Rec&a,const Rec&q,const Hit&h,int rank,float margin){
 array<float,NF> f{};auto an=toks(a.c),qn=toks(q.c),aa=toks(a.a),qa=toks(q.a),ap=toks(a.p),qp=toks(q.p);auto anum=numbers(a.a),qnum=numbers(q.a);int ni=intersection(an,qn),ai=intersection(aa,qa),nu=intersection(anum,qnum);
 f[0]=edit(a.c,q.c);f[1]=dice(grams(a.c),grams(q.c));f[2]=edit(joined(an),joined(qn));f[3]=div0(ni,an.size()+qn.size()-ni);f[4]=div0(ni,min(an.size(),qn.size()));f[5]=div0(ni,max(an.size(),qn.size()));f[6]=a.c==q.c;f[7]=compact(a.c)==compact(q.c);f[8]=edit(a.n,q.n);f[9]=leng(a.c,q.c);f[10]=weighted_overlap(idx,a.country,1,an,qn);f[11]=edit(a.p,q.p);f[12]=div0(intersection(ap,qp),min(ap.size(),qp.size()));
 f[13]=edit(a.a,q.a);f[14]=dice(grams(a.a),grams(q.a));f[15]=div0(ai,aa.size()+qa.size()-ai);f[16]=div0(ai,min(aa.size(),qa.size()));f[17]=div0(ai,max(aa.size(),qa.size()));f[18]=weighted_overlap(idx,a.country,3,aa,qa);f[19]=div0(nu,anum.size()+qnum.size()-nu);f[20]=div0(nu,min(anum.size(),qnum.size()));f[21]=!anum.empty()&&!qnum.empty();string af=firstnum(a.a),qf=firstnum(q.a);f[22]=!af.empty()&&af==qf;f[23]=!af.empty()&&!qf.empty()&&af!=qf;f[24]=q.missing;f[25]=a.script;f[26]=q.script;f[27]=an.size();f[28]=qn.size();f[29]=aa.size();f[30]=qa.size();auto aw=split(a.c),qw=split(q.c);f[31]=aw.empty()||qw.empty()?0:edit(aw[0],qw[0]);
 float sum=0,mn=1,mx=0;for(auto&w:qn){float best=0;for(auto&z:an)best=max(best,edit(w,z));sum+=best;mn=min(mn,best);mx=max(mx,best);}f[32]=div0(sum,qn.size());f[33]=qn.empty()?0:mn;f[34]=mx;f[35]=leng(a.a,q.a);f[36]=!a.a.empty()&&a.a==q.a;f[37]=f[1]*f[14];f[38]=min(f[1],f[14]);f[39]=h.ns;f[40]=h.as;f[41]=h.rank;f[42]=rank;f[43]=margin;auto ax=split(a.a),qx=split(q.a);f[44]=!ax.empty()&&!qx.empty()&&ax.front()==qx.front();f[45]=!ax.empty()&&!qx.empty()&&ax.back()==qx.back();f[46]=ni;f[47]=ai;f[48]=q.eid.rfind("S3-",0)==0;f[49]=a.c.empty()||q.c.empty();f[50]=compact(a.p)==compact(q.p);f[51]=float(anum.size())-float(qnum.size());auto z=idx.inv.find(idx.key(a.country,0,compact(q.c)));f[52]=z==idx.inv.end()?0:log1p(z->second.df);auto ar=toks(a.n),qr=toks(q.n);int ri=intersection(ar,qr);f[53]=div0(ri,ar.size()+qr.size()-ri);return f;
}
struct OutRow{uint32_t tid,aid;int32_t owner,tfold,afold;array<float,NF> f;};
static_assert(sizeof(OutRow)==20+4*NF);
struct WorkItem{uint32_t tid;Rec r;};
int main(int argc,char**argv){
 if(argc<6){cerr<<"usage: retrieve ANCHORS TARGETS OUTPUT_DIR THREADS TOPK [LIMIT] [OFFSET]\n";return 2;}
 string anchorfile=argv[1],targetfile=argv[2],outdir=argv[3];int threads=stoi(argv[4]),topk=stoi(argv[5]);uint64_t limit=argc>6?stoull(argv[6]):0,offset=argc>7?stoull(argv[7]):0;
 filesystem::create_directories(outdir);ofstream schema(outdir+"/features.json");schema<<"[";for(size_t i=0;i<FN.size();i++)schema<<(i?",":"")<<"\""<<FN[i]<<"\"";schema<<"]";schema.close();
 Index idx;idx.build(anchorfile);ifstream input(targetfile);if(!input)throw runtime_error("target file missing");mutex readlock,loglock;uint64_t counter=0;atomic<uint64_t> done{0},pairs{0},truth{0},retrieved{0};auto start=chrono::steady_clock::now();vector<thread> workers;
 for(int threadid=0;threadid<threads;threadid++)workers.emplace_back([&,threadid]{
  Searcher search(idx);ofstream output(outdir+"/pairs-"+to_string(threadid)+".bin",ios::binary);string line;vector<WorkItem> batch;batch.reserve(512);
  while(true){batch.clear();{
   lock_guard<mutex> guard(readlock);while(batch.size()<512&&(!limit||counter<limit)&&getline(input,line)){batch.push_back({uint32_t(counter+offset),parse(line)});counter++;}
  }if(batch.empty())break;
  for(auto&w:batch){auto& q=w.r;auto hits=search.retrieve(q,24);auto qng=grams(q.c),qag=grams(q.a);auto qnt=toks(q.c),qat=toks(q.a);
   // This deterministic, label-free filter is part of blocking; every survivor is scored.
   for(auto&h:hits){auto&a=idx.anchors[h.aid];float n=dice(grams(a.c),qng),ad=dice(grams(a.a),qag);auto at=toks(a.c);float contain=div0(intersection(at,qnt),min(at.size(),qnt.size()));float num=!firstnum(q.a).empty()&&firstnum(a.a)==firstnum(q.a);h.nq=n;h.aq=ad;h.rank=.50f*n+.25f*ad+.18f*min(n,ad)+.10f*contain+.07f*num;if(a.c==q.c)h.rank+=.08f;if(q.missing)h.rank=.8f*n+.2f*contain;}
   sort(hits.begin(),hits.end(),[](auto a,auto b){if(a.rank==b.rank)return a.aid<b.aid;return a.rank>b.rank;});auto alternatives=hits;if((int)hits.size()>topk)hits.resize(topk);
   // Protect complementary retrieval routes from domination by the name-heavy filter.
   auto append_unique=[&](const Hit&h){if(none_of(hits.begin(),hits.end(),[&](auto z){return z.aid==h.aid;}))hits.push_back(h);};
   if(!q.missing){sort(alternatives.begin(),alternatives.end(),[](auto a,auto b){return a.aq==b.aq?a.aid<b.aid:a.aq>b.aq;});for(int k=0;k<min(2,(int)alternatives.size());k++)if(alternatives[k].aq>.35f)append_unique(alternatives[k]);}
   sort(alternatives.begin(),alternatives.end(),[](auto a,auto b){return a.nq==b.nq?a.aid<b.aid:a.nq>b.nq;});for(int k=0;k<min(2,(int)alternatives.size());k++)if(alternatives[k].nq>.60f)append_unique(alternatives[k]);
   bool found=false;float margin=hits.size()>1?hits[0].rank-hits[1].rank:hits.empty()?0:hits[0].rank;
   for(size_t j=0;j<hits.size();j++){auto&h=hits[j];if((int32_t)h.aid==q.owner)found=true;OutRow row{w.tid,h.aid,q.owner,q.fold,idx.anchors[h.aid].fold,features(idx,idx.anchors[h.aid],q,h,j,margin)};output.write((char*)&row,sizeof(row));}
   pairs+=hits.size();if(q.owner>=0){truth++;if(found)retrieved++;}
  }
  auto d=(done+=batch.size());if(d%102400<512){lock_guard<mutex> guard(loglock);double sec=chrono::duration<double>(chrono::steady_clock::now()-start).count();cerr<<"QUERY "<<d<<" pairs="<<pairs<<" sec="<<sec<<" qps="<<d/sec<<" link_recall="<<(truth?double(retrieved)/truth:0)<<"\n";}
  }
 });for(auto&t:workers)t.join();
 ofstream report(outdir+"/run.json");report<<"{\"queries\":"<<done<<",\"pairs\":"<<pairs<<",\"true_links\":"<<truth<<",\"retrieved_links\":"<<retrieved<<",\"topk\":"<<topk<<",\"seconds\":"<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<"}";
 cerr<<"DONE queries="<<done<<" pairs="<<pairs<<" link_recall="<<(truth?double(retrieved)/truth:0)<<"\n";
 return 0;
}
