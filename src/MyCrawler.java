import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Set;
//import java.util.regex.Pattern;

import edu.uci.ics.crawler4j.crawler.*;
//import edu.uci.ics.crawler4j.parser.*;
import edu.uci.ics.crawler4j.url.*;


public class MyCrawler extends WebCrawler {
	//private final static Pattern FILTERS = Pattern.compile(".*(\\.(css|js|gif|jpg|png|mp3|zip|gz))$");
	private final static String seedurl1 = "https://steamcommunity.com/";
	private final static String seedurl2 = "http://steamcommunity.com/";
	//private final static String appid = "755790";
	//private final static String appid = "1330123889";
	private final static String appid = null;
	//private final static String seedurl1 = "https://sensortower.com/ios/US/";
	//private final static String seedurl2 = "http://sensortower.com/ios/US/";

	@Override
	public boolean shouldVisit(Page referringPage, WebURL url){
		String href = url.getURL().toLowerCase();
		//System.out.println(href);
		if(!(href.startsWith(seedurl1) || href.startsWith(seedurl2)))
			return false;
		//if(appid != null && !(href.contains(appid)))
		//	return false;
		if(!(referringPage == null || referringPage.getContentType() == null || referringPage.getContentType().contains("html")))
			return false;
		if(href.contains("login") || href.contains("forum")|| href.contains("market") || href.contains("?l="))
			return false;
		if(href.contains("png")||href.contains("jpg"))
			return false;
		if(!href.contains("steam"))
			return false;
		//if(!(href.contains("recommended")||href.contains("reviews")&&!(href.contains("login"))))
		//	return false;
		return true;
		//return ((href.startsWith(seedurl1) || href.startsWith(seedurl2)) 
		//		&& (referringPage == null || referringPage.getContentType() == null || referringPage.getContentType().contains("html") 
		//		|| referringPage.getContentType().contains("doc") || referringPage.getContentType().contains("pdf") || referringPage.getContentType().contains("image")));
	}
	private final static String output1 = "data/hreflog.csv";
	private final static String output2 = "data/result.csv";
	private final static String output3 = "data/test.csv";
	private static FileWriter out1 = null;
	private static FileWriter out2 = null;
	private static FileWriter out3 = null;
	private static Set<WebURL> UniquePages = null;
	private static int totaloutURL = 0;
	public MyCrawler(){
		try {
			out1 = new FileWriter(output1);
			out2 = new FileWriter(output2);
			out3 = new FileWriter(output3);
			totaloutURL = 0;
			UniquePages = new HashSet<WebURL>();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	@Override
	public void onBeforeExit(){
		try {
			out1.flush();
			out2.flush();
			out3.flush();
			System.out.print("Total outgoing URL: "+totaloutURL);

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	@Override
    protected WebURL handleUrlBeforeProcess(WebURL curURL) {
		String url = curURL.getURL();
		if(/*url.startsWith("https://steamcommunity.com/market/search")*/ !url.contains("?p="))
			url = url.split("\\?")[0];
		if(url.endsWith("/"))
			url = url.substring(0, url.length()-1);
		if(url.startsWith("https://steamcommunity.com/id/") && url.split("/").length == 5){
			url += "/recommended";
		}
		if(url.startsWith("https://steamcommunity.com/profiles/") && url.split("/").length == 5){
			url += "/recommended";
		}
		if(url.startsWith("https://steamcommunity.com/app/") && url.split("/").length == 5){
			url += "/reviews";
		}
		//System.out.println(curURL.getURL()+' '+url);
		curURL.setURL(url);
        return curURL;
    }
	@Override
	protected void handlePageStatusCode(WebURL webUrl, int statusCode, String statusDescription) {
		try {
			System.out.println(webUrl.getURL()+"\t"+statusCode);
			out1.write(webUrl.getURL()+","+statusCode+"\n");
			if(statusCode / 100 == 4){
				Thread.sleep(10000);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
	@Override
	public void visit(Page page){
		String url = page.getWebURL().getURL();
		int scode = page.getStatusCode();
		//System.out.println(scode);
		//Set<WebURL> links = page.getParseData().getOutgoingUrls();
		try {
			//out1.write(url+","+scode+"\n");
			if(scode / 100 == 2){
				//out2.write(url+","+page.getContentData().length+","+links.size()+","+page.getContentType().split(";")[0]+"\n");
				if(!url.contains("recommended"))
					return ;
				String content = new String(page.getContentData(), StandardCharsets.UTF_8);
				String[] ss = content.split("\n");
				String s1 = "0";
				String s2 = "no review";
				for(String s: ss){
					if(s.contains("icon_thumbsUp.png"))
						s1 = "1";
					if(s.contains("icon_thumbsDown.png"))
						s1 = "-1";
					if(s.contains("ReviewText"))
						s2 = s;
				}
				if(!s1.equals("0") && !s2.equals("no review")){
					s2 = s2.trim();
					StringBuffer bs2 = new StringBuffer("");
					String[] ss2 = s2.split("[<>]");
					int l = ss2.length;
					for(int i=0;i<l;i+=2)
						bs2.append(ss2[i]+" ");
					out2.write(url+","+s1+","+bs2.toString()+"\n");
				}
			}

			out1.flush();
			out2.flush();
			out3.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
