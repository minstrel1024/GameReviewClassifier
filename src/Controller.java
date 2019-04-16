import edu.uci.ics.crawler4j.crawler.*;
import edu.uci.ics.crawler4j.fetcher.*;
import edu.uci.ics.crawler4j.robotstxt.*;


public class Controller {
	public static void main(String[] args) throws Exception{
		String crawlStorageFolder = "data/crawl";
		int numberOfCrawlers = 7;
		int maxPages = 200000;
		int maxDepth = 200;
		//String seed = "https://steamcommunity.com/app/755790/reviews/?filterLanguage=english";
		//String seed = "https://sensortower.com/ios/US/tencent-mobile-international-limited/app/pubg-mobile/1330123889/review-history";
		//String seed = "https://steamcommunity.com/profiles/76561198007833990/recommended/";
		String seed = "https://steamcommunity.com/app/578080/reviews/"; //PUBG
		//String seed = "https://steamcommunity.com/app/583950/reviews/"; //Artifact
		//String seed = "https://steamcommunity.com/app/289070/reviews/"; //CIV6
		//String seed = "https://steamcommunity.com/app/570/reviews/"; //Dota2
		
		CrawlConfig config = new CrawlConfig();
		config.setCrawlStorageFolder(crawlStorageFolder);
		config.setMaxDepthOfCrawling(maxDepth);
		config.setMaxPagesToFetch(maxPages);
		PageFetcher pageFetcher = new PageFetcher(config);
		RobotstxtConfig robotstxtConfig = new RobotstxtConfig();
		RobotstxtServer robotstxtServer = new RobotstxtServer(robotstxtConfig, pageFetcher);
		CrawlController controller = new CrawlController(config, pageFetcher, robotstxtServer);
		
		controller.addSeed(seed);
		controller.start(MyCrawler.class, numberOfCrawlers);
		
	}
}
